"""Error taxonomy and answer flips for quantized models (RTN g128).

For each (model, task, bits) we count, among items that were correct (score>=0.5) in bf16:
  - lost: now wrong; split into
      degenerate: output hit max_tokens with low lexical diversity (distinct-2 < 0.5) or is empty
      truncated : hit max_tokens but not degenerate (rambling / over-long reasoning)
      wrong     : terminated normally but wrong answer (fluent error)
and among items wrong in bf16 how many became correct ("gained") — the noise floor of flips.
Writes results/error_taxonomy.csv and prints a compact table + example failures.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    df = pd.read_csv(ROOT / "results" / "item_scores.csv")
    df = df[(df.method == "rtn") & (df.group == 128) & (df.task != "creative")]
    base = df[df.bits == 16].set_index(["model", "task", "id"]).score
    rows = []
    for (mo, t, b), d in df[df.bits != 16].groupby(["model", "task", "bits"]):
        d = d.set_index("id")
        b0 = base.loc[(mo, t)].reindex(d.index)
        was = b0 >= 0.5
        now = d.score >= 0.5
        lost = d[was & ~now]
        degen = (lost.truncated & (lost.distinct2 < 0.5)) | (lost.n_tokens == 0)
        trunc = lost.truncated & ~degen
        rows.append(dict(model=mo, task=t, bits=b, n_correct_bf16=int(was.sum()), lost=len(lost),
                         lost_degenerate=int(degen.sum()), lost_truncated=int(trunc.sum()),
                         lost_wrong_fluent=int(len(lost) - degen.sum() - trunc.sum()),
                         gained=int((~was & now).sum()), n_wrong_bf16=int((~was).sum()),
                         flip_rate=float(((was != now)).mean())))
    r = pd.DataFrame(rows)
    r["frac_lost"] = r.lost / r.n_correct_bf16.clip(lower=1)
    r["frac_lost_fluent"] = r.lost_wrong_fluent / r.lost.clip(lower=1)
    r.to_csv(ROOT / "results" / "error_taxonomy.csv", index=False)
    agg = r.groupby(["task", "bits"])[["n_correct_bf16", "lost", "lost_degenerate", "lost_truncated",
                                       "lost_wrong_fluent", "gained"]].sum()
    agg["frac_lost"] = (agg.lost / agg.n_correct_bf16).round(3)
    agg["share_fluent_errors"] = (agg.lost_wrong_fluent / agg.lost.clip(lower=1)).round(2)
    print(agg.to_string())
    agg.to_csv(ROOT / "results" / "error_taxonomy_by_task_bits.csv")

    # A few concrete examples: 4-bit fluent GSM8K/TriviaQA errors where bf16 was right.
    suite = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "datasets" / "suite.jsonl")}
    ex_out = []
    for mo in sorted(df.model.unique()):
        g16 = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "results" / "gens" / f"{mo}_b16_g128.jsonl")}
        p4 = ROOT / "results" / "gens" / f"{mo}_b4_g128.jsonl"
        if not p4.exists():
            continue
        g4 = {json.loads(l)["id"]: json.loads(l) for l in open(p4)}
        sc = df[(df.model == mo)].set_index(["bits", "id"]).score
        for t in ("factqa", "arith"):
            ids = [i for i in g16 if g16[i]["task"] == t and sc.get((16, i), 0) >= 0.5 and sc.get((4, i), 1) < 0.5][:2]
            for i in ids:
                ex_out.append(dict(model=mo, task=t, id=i, question=suite[i]["messages"][0]["content"][-200:],
                                   gold=str(suite[i]["gold"])[:80], bf16=g16[i]["response"][-160:], int4=g4[i]["response"][-160:]))
    json.dump(ex_out, open(ROOT / "results" / "error_examples_int4.json", "w"), indent=1)
    print(f"{len(ex_out)} examples -> results/error_examples_int4.json")


if __name__ == "__main__":
    main()
