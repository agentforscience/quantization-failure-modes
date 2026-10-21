"""Task-specific token-level KL(bf16 || quantized), top-1 agreement, output NLL and WikiText-2 PPL.

For each task we teacher-force the bf16 model's own greedy responses (first N items per task,
response truncated to 256 tokens) and compare next-token distributions on response tokens.
The bf16 distribution is stored as top-K (K=128) log-probs; KL is computed over those K tokens
plus one lumped "tail" bucket. By the data-processing inequality this is a lower bound on the
exact KL; `--validate` measures the gap against exact KL on a few sequences.

Usage: python src/kl.py --model qwen1.5b [--levels 16,8,6,5,4,3,2] [--group 128]
Output: results/kl/{model}_g{group}.csv  (rows: level x task, incl. task="wikitext")
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models import MODELS  # noqa: E402
from quant import quantizable, rtn_into_  # noqa: E402

K = 128
TASKS = ["arith", "factqa", "reading", "code", "instr", "creative"]


def build_sequences(tok, model_key, n_per_task, max_resp=256):
    suite = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "datasets" / "suite.jsonl")}
    gens = [json.loads(l) for l in open(ROOT / "results" / "gens" / f"{model_key}_b16_g128.jsonl")]
    seqs, cnt = [], {t: 0 for t in TASKS}
    for g in gens:
        if cnt[g["task"]] >= n_per_task:
            continue
        p_ids = tok.apply_chat_template(suite[g["id"]]["messages"], add_generation_prompt=True, tokenize=True)
        if isinstance(p_ids, dict) or hasattr(p_ids, "input_ids"):
            p_ids = p_ids["input_ids"]
        r_ids = tok(g["response"], add_special_tokens=False)["input_ids"][:max_resp]
        if len(r_ids) < 1:
            continue
        cnt[g["task"]] += 1
        seqs.append(dict(task=g["task"], ids=torch.tensor(p_ids + r_ids), n_resp=len(r_ids)))
    return seqs


def wikitext_windows(tok, n_windows, win=1024):
    text = open(ROOT / "datasets" / "wikitext2_test.txt").read()  # WikiText-2 raw test split
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [dict(task="wikitext", ids=torch.tensor(ids[i * win:(i + 1) * win]), n_resp=win - 1) for i in range(n_windows)]


@torch.no_grad()
def resp_logprobs(model, s, dev):
    """Log-probs (fp32) predicting each of the last n_resp tokens: shape [n_resp, V]."""
    ids = s["ids"].to(dev)[None]
    out = model(ids, logits_to_keep=s["n_resp"] + 1)
    logits = out.logits[0, :-1].float()  # positions predicting the last n_resp tokens
    return F.log_softmax(logits, -1), ids[0, -s["n_resp"]:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", default="16,8,6,5,4,3,2")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--n_per_task", type=int, default=60)
    ap.add_argument("--wiki_windows", type=int, default=32)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--max_gb_per_gpu", type=float, default=0, help=">0: shard model over all visible GPUs")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    out_path = ROOT / "results" / "kl" / f"{args.model}_g{args.group}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    repo = MODELS[args.model][0]
    tok = AutoTokenizer.from_pretrained(repo)
    if args.max_gb_per_gpu > 0:  # pipeline-shard over several small GPU slices
        mm = {i: f"{args.max_gb_per_gpu}GiB" for i in range(torch.cuda.device_count())}
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto", max_memory=mm).eval()
        dev = model.get_input_embeddings().weight.device
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to(dev).eval()
    seqs = build_sequences(tok, args.model, args.n_per_task) + wikitext_windows(tok, args.wiki_windows)
    print(f"{len(seqs)} sequences, {sum(s['n_resp'] for s in seqs)} scored tokens", flush=True)
    orig = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if quantizable(n, p)}
    print(f"{len(orig)} quantizable matrices", flush=True)

    # Reference pass (bf16): store top-K log-probs and argmax per position.
    ref = []
    exact_ref = []
    for i, s in enumerate(seqs):
        lp, tgt = resp_logprobs(model, s, dev)
        v, ix = lp.topk(K, -1)
        ref.append((v.cpu(), ix.cpu()))
        if args.validate and i % 50 == 0 and len(exact_ref) < 6:
            exact_ref.append((i, lp.cpu()))

    rows = []
    for b in [int(x) for x in args.levels.split(",")]:
        t0 = time.time()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in orig:
                    rtn_into_(p.data, orig[n], b, args.group)
        acc = {}
        for i, s in enumerate(seqs):
            lq, tgt = resp_logprobs(model, s, dev)
            ldev = lq.device; tgt = tgt.to(ldev)
            pv, pix = ref[i][0].to(ldev), ref[i][1].to(ldev)
            p = pv.exp()
            qv = lq.gather(1, pix)
            r_p = (1 - p.sum(1)).clamp(min=1e-12)
            r_q = (1 - qv.exp().sum(1)).clamp(min=1e-12)
            kl_tok = (p * (pv - qv)).sum(1) + r_p * (r_p.log() - r_q.log())
            kl_tok = kl_tok.clamp(min=0)
            nll = -lq.gather(1, tgt[:, None])[:, 0]
            agree = (lq.argmax(1) == pix[:, 0]).float()
            a = acc.setdefault(s["task"], dict(kl=0.0, nll=0.0, agree=0.0, n=0, seq_kl=[], kl_max=[]))
            a["kl"] += kl_tok.sum().item(); a["nll"] += nll.sum().item(); a["agree"] += agree.sum().item()
            a["n"] += len(kl_tok); a["seq_kl"].append(kl_tok.mean().item())
            if args.validate:
                for j, lp_exact in exact_ref:
                    if j == i:
                        ex = (lp_exact.exp() * (lp_exact - lq.cpu())).sum(1).mean().item()
                        print(f"  validate bits={b} seq{i}: exact KL={ex:.4f} topK-bound={kl_tok.mean().item():.4f}")
        for t, a in acc.items():
            rows.append(dict(model=args.model, bits=b, group=args.group, task=t, kl=a["kl"] / a["n"],
                             kl_seqmean=sum(a["seq_kl"]) / len(a["seq_kl"]), top1_agree=a["agree"] / a["n"],
                             nll=a["nll"] / a["n"], ppl=math.exp(min(50, a["nll"] / a["n"])), n_tokens=a["n"],
                             n_seqs=len(a["seq_kl"])))
        print(f"[{args.model}] bits={b} ({time.time()-t0:.0f}s): " +
              " ".join(f"{r['task']}:{r['kl']:.3f}" for r in rows if r["bits"] == b), flush=True)
        pd.DataFrame(rows).to_csv(out_path.with_suffix(".partial"), index=False)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    out_path.with_suffix(".partial").unlink()


if __name__ == "__main__":
    main()
