"""Generate greedy outputs for the whole task suite at several RTN weight precisions.

One vLLM engine is created per model; between precision levels the transformer linear
weights are restored from a CPU backup and re-quantized in place inside the worker
(prefix caching disabled so no KV computed under other weights is reused).

Usage: python src/generate.py --model qwen1.5b --levels 16,8,6,5,4,3,2 --gpu_gb 5
Outputs: results/gens/{model}_b{bits}_g{group}.jsonl  (one line per suite item)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src")
sys.path.insert(0, SRC)
from models import MODELS  # noqa: E402


def _quantize_worker(self, bits, group, src):
    """Runs inside the vLLM worker: restore bf16 weights, then RTN-quantize to `bits`."""
    import sys
    import torch
    sys.path.insert(0, src)
    from quant import quantizable, rtn_into_
    model = self.model_runner.model
    if not hasattr(self, "_qorig"):
        self._qorig = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if quantizable(n, p)}
        # TP shards must keep group boundaries aligned with the unsharded matrix
        bad = [n for n, w in self._qorig.items() if group > 0 and w.shape[1] % group]
        assert not bad, f"input dim not divisible by group under this TP size: {bad[:3]}"
    n_q = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in self._qorig:
                rtn_into_(p.data, self._qorig[n], bits, group)
                n_q += 1
    torch.cuda.synchronize()
    return n_q, sorted(self._qorig)[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", default="16,8,6,5,4,3,2")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--gpu_gb", type=float, default=12.0)
    ap.add_argument("--max_num_seqs", type=int, default=128)
    ap.add_argument("--batched_tokens", type=int, default=4096)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (shard over small GPU slices)")
    ap.add_argument("--limit", type=int, default=0, help="debug: items per task")
    args = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams

    repo, _, _ = MODELS[args.model]
    is_awq = args.model.endswith("-awq")
    levels = [4] if is_awq else [int(x) for x in args.levels.split(",")]
    outdir = ROOT / "results" / "gens"
    outdir.mkdir(parents=True, exist_ok=True)

    def outpath(b):
        tag = "awq" if is_awq else f"b{b}_g{args.group}"
        return outdir / f"{args.model}_{tag}.jsonl"

    todo = [b for b in levels if not outpath(b).exists()]
    if not todo:
        print("nothing to do"); return

    suite = [json.loads(l) for l in open(ROOT / "datasets" / "suite.jsonl")]
    if args.limit:
        from collections import defaultdict
        cnt = defaultdict(int); keep = []
        for it in suite:
            if cnt[it["task"]] < args.limit:
                keep.append(it); cnt[it["task"]] += 1
        suite = keep

    # total GPU memory via nvidia-smi: avoids creating a CUDA context (~0.8 GB) in this parent process
    import subprocess
    total = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])) * 2**20
    llm = LLM(model=repo, dtype="bfloat16" if not is_awq else "auto", seed=0,
              gpu_memory_utilization=args.gpu_gb * 2**30 / total, max_model_len=2560,
              max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.batched_tokens,
              enforce_eager=args.eager, enable_prefix_caching=False, trust_remote_code=False,
              tensor_parallel_size=args.tp)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True) for it in suite]
    sps = [SamplingParams(temperature=0.0, max_tokens=it["max_tokens"], seed=0) for it in suite]

    for b in todo:
        t0 = time.time()
        if not is_awq:
            info = llm.collective_rpc(_quantize_worker, args=(b, args.group, SRC))
            print(f"[{args.model}] bits={b}: quantized {[i[0] for i in info]} matrices/worker e.g. {info[0][1]}", flush=True)
            assert all(i[0] > 0 for i in info)
        outs = llm.generate(prompts, sps, use_tqdm=False)
        tmp = outpath(b).with_suffix(".partial")
        with open(tmp, "w") as f:
            for it, o in zip(suite, outs):
                c = o.outputs[0]
                f.write(json.dumps({"task": it["task"], "id": it["id"], "model": args.model,
                                    "bits": 4 if is_awq else b, "method": "awq" if is_awq else "rtn",
                                    "group": args.group, "response": c.text, "n_tokens": len(c.token_ids),
                                    "finish": c.finish_reason}) + "\n")
        if args.limit:
            print(outs[0].outputs[0].text[:500]); tmp.unlink(); continue
        tmp.rename(outpath(b))
        print(f"[{args.model}] bits={b} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
