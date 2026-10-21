"""LLM-as-judge for the creative-writing task.

Judge: Qwen3-8B-AWQ (thinking disabled), greedy. Each story is rated 1-10 on fluency,
coherence, relevance-to-prompt and creativity; the item score is the mean / 10.
The judge sees only the writing prompt and the story (blind to model and precision).
Results cached in results/judge_cache.jsonl keyed by (model, method, bits, group, id).

Usage: python src/judge.py --gpu_gb 11
"""
import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
JUDGE = "Qwen/Qwen3-8B-AWQ"
CACHE = ROOT / "results" / "judge_cache.jsonl"

TEMPLATE = """You are an expert creative-writing editor. Rate the short story below, written in response to the given prompt.

Writing prompt: {prompt}

Story:
<<<
{story}
>>>

Rate each criterion on an integer scale from 1 (very poor) to 10 (excellent):
- fluency: grammatical, readable English prose (gibberish, broken or repeated text = 1)
- coherence: the story is logically consistent and has a clear progression
- relevance: the story follows the writing prompt
- creativity: originality, vividness and engagement

Reply with only a JSON object: {{"fluency": n, "coherence": n, "relevance": n, "creativity": n}}"""

KEYS = ["fluency", "coherence", "relevance", "creativity"]


def key(g):
    return f"{g['model']}|{g['method']}|{g['bits']}|{g['group']}|{g['id']}"


def parse(text):
    out = {}
    for k in KEYS:
        m = re.search(rf'"{k}"\s*:\s*(\d+)', text)
        if m:
            out[k] = min(10, max(1, int(m.group(1))))
    return out if len(out) == len(KEYS) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_gb", type=float, default=11)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()
    import torch
    from vllm import LLM, SamplingParams

    suite = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "datasets" / "suite.jsonl")}
    done = {}
    if CACHE.exists():
        for l in open(CACHE):
            r = json.loads(l); done[r["key"]] = r
    todo = []
    for p in sorted((ROOT / "results" / "gens").glob("*.jsonl")):
        for l in open(p):
            g = json.loads(l)
            if g["task"] == "creative" and key(g) not in done:
                todo.append(g)
    print(f"{len(todo)} stories to judge", flush=True)
    if not todo:
        return
    # total GPU memory via nvidia-smi: avoids creating a CUDA context (~0.8 GB) in this parent process
    import subprocess
    total = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])) * 2**20
    llm = LLM(JUDGE, gpu_memory_utilization=args.gpu_gb * 2**30 / total, max_model_len=3072, max_num_seqs=64,
              enable_prefix_caching=True, seed=0, tensor_parallel_size=args.tp, enforce_eager=True,
              max_num_batched_tokens=2048)
    tok = llm.get_tokenizer()
    prompts = []
    for g in todo:
        wp = suite[g["id"]]["messages"][0]["content"].split("\n\n", 1)[1]
        story = g["response"].strip()[:4000] or "(empty)"
        msg = [{"role": "user", "content": TEMPLATE.format(prompt=wp, story=story)}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=60), use_tqdm=False)
    n_fail = 0
    with open(CACHE, "a") as f:
        for g, o in zip(todo, outs):
            sc = parse(o.outputs[0].text)
            n_fail += sc is None
            f.write(json.dumps({"key": key(g), "scores": sc, "raw": o.outputs[0].text}) + "\n")
    print(f"done; parse failures: {n_fail}")


if __name__ == "__main__":
    main()
