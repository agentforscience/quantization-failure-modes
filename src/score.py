"""Score every generation file in results/gens/ (all tasks except creative-writing quality,
which is judged by src/judge.py). Writes results/item_scores.csv with one row per
(model, method, bits, group, task, id) and a `score` column in [0, 1].

Scoring rules
- arith  : GSM8K; number after "The answer is" (fallback: last number) == gold.
- factqa : TriviaQA; normalized prediction equals, or contains as a whole phrase, any gold alias.
- reading: SQuAD token-F1 (max over gold spans); EM kept as `score2`.
- code   : HumanEval pass@1; extracted code + unit tests executed in a subprocess (10 s timeout).
- instr  : IFEval prompt-level strict accuracy; instruction-level accuracy kept as `score2`.
- creative: placeholder score = NaN here; degeneration stats (distinct-2, repetition) stored.
"""
import collections
import json
import os
import re
import string
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from instruction_following_eval import evaluation_lib  # noqa: E402
from absl import logging as absl_logging  # noqa: E402
absl_logging.set_verbosity(absl_logging.FATAL)

SUITE = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "datasets" / "suite.jsonl")}
CACHE = ROOT / "results" / "score_cache"


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def score_arith(resp, gold):
    m = re.findall(r"answer is[:\s\$\\\(\)boxed\{]*(-?[\d,]*\.?\d+)", resp, flags=re.I)
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", resp)
    if not m:
        return 0.0
    try:
        return float(abs(float(m[-1].replace(",", "")) - float(gold)) < 1e-6)
    except ValueError:
        return 0.0


def score_factqa(resp, aliases):
    p = norm(resp.strip().split("\n")[0])
    if not p:
        return 0.0
    for a in aliases:
        if a and (p == a or re.search(r"\b" + re.escape(a) + r"\b", p)):
            return 1.0
    return 0.0


def f1(pred, gold):
    pt, gt = norm(pred).split(), norm(gold).split()
    common = collections.Counter(pt) & collections.Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(pt), ns / len(gt)
    return 2 * pr * rc / (pr + rc)


def score_reading(resp, golds):
    pred = resp.strip().split("\n")[0]
    return max(f1(pred, g) for g in golds), float(max(norm(pred) == norm(g) for g in golds))


def extract_code(resp, gold):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", resp, flags=re.S)
    if not blocks:  # unterminated block
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*)", resp, flags=re.S)
    code = max(blocks, key=len) if blocks else resp
    if f"def {gold['entry_point']}" not in code:
        code = gold["prompt"] + code
    # keep the prompt's imports available
    header = "\n".join(l for l in gold["prompt"].split("\n") if l.startswith(("import ", "from ")))
    return header + "\n" + code


def run_code(code, gold):
    prog = code + "\n\n" + gold["test"] + f"\n\ncheck({gold['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "prog.py")
        open(fn, "w").write(prog)
        try:
            r = subprocess.run([sys.executable, "-I", fn], cwd=d, capture_output=True, timeout=10)
            return float(r.returncode == 0)
        except subprocess.TimeoutExpired:
            return 0.0


def score_instr(resp, gold):
    kw = [{k: v for k, v in d.items() if v is not None} for d in gold["kwargs"]]
    inp = evaluation_lib.InputExample(key=gold["key"], instruction_id_list=gold["instruction_id_list"],
                                      prompt=gold["prompt"], kwargs=kw)
    out = evaluation_lib.test_instruction_following_strict(inp, {gold["prompt"]: resp})
    fl = out.follow_instruction_list
    return float(out.follow_all_instructions), sum(fl) / len(fl)


def degeneration(resp):
    toks = resp.split()
    bigr = list(zip(toks, toks[1:]))
    distinct2 = len(set(bigr)) / max(1, len(bigr))
    return distinct2


def score_file(path):
    cache = CACHE / (path.stem + ".csv")
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        return pd.read_csv(cache)
    rows, code_jobs = [], []
    for line in open(path):
        g = json.loads(line)
        it = SUITE[g["id"]]
        t, resp, gold = g["task"], g["response"], it["gold"]
        s2 = float("nan")
        if t == "arith":
            s = score_arith(resp, gold)
        elif t == "factqa":
            s = score_factqa(resp, gold)
        elif t == "reading":
            s, s2 = score_reading(resp, gold)
        elif t == "instr":
            try:
                s, s2 = score_instr(resp, gold)
            except Exception as e:  # checker crash on garbage output => not followed
                print("ifeval error", g["id"], repr(e)[:100]); s, s2 = 0.0, 0.0
        elif t == "code":
            s = None; code_jobs.append((len(rows), extract_code(resp, gold), gold))
        else:
            s = float("nan")
        rows.append(dict(model=g["model"], method=g["method"], bits=g["bits"], group=g["group"], task=t, id=g["id"],
                         score=s, score2=s2, n_tokens=g["n_tokens"], truncated=g["finish"] == "length",
                         distinct2=degeneration(resp)))
    with ThreadPoolExecutor(24) as ex:
        res = list(ex.map(lambda j: run_code(j[1], j[2]), code_jobs))
    for (i, _, _), r in zip(code_jobs, res):
        rows[i]["score"] = r
    df = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def main():
    files = sorted((ROOT / "results" / "gens").glob("*.jsonl"))
    dfs = []
    for p in files:
        df = score_file(p)
        dfs.append(df)
        summ = df.groupby("task").score.mean().round(3).to_dict()
        print(p.stem, summ, flush=True)
    pd.concat(dfs).to_csv(ROOT / "results" / "item_scores.csv", index=False)


if __name__ == "__main__":
    main()
