"""Render markdown tables for REPORT.md from results/*.csv -> results/tables.md."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
import sys
sys.path.insert(0, str(ROOT / "src"))
from models import MODELS, LEVELS  # noqa: E402

TASKS = ["arith", "factqa", "code", "instr", "reading", "creative"]


def md(df, floatfmt="{:.2f}"):
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        vals = [("" if pd.isna(v) else (floatfmt.format(v) if isinstance(v, float) else str(v))) for v in row]
        lines.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join(lines)


def main():
    s = pd.read_csv(RES / "summary_scores.csv")
    out = []
    # Table 1: absolute bf16 scores with CI + AWQ
    b = s[(s.bits == 16) & (s.group == 128)].copy()
    b["cell"] = b.apply(lambda r: f"{r.score:.3f} [{r.ci_lo:.2f},{r.ci_hi:.2f}]", axis=1)
    t1 = b.pivot(index="model", columns="task", values="cell").reindex(columns=TASKS)
    out.append("### Table A. bf16 baseline scores (95% bootstrap CI)\n\n" + md(t1))
    # Table 2: retained by model x bits for each task
    sw = s[(s.method == "rtn") & (s.group == 128)]
    for t in TASKS:
        p = sw[sw.task == t].pivot(index="model", columns="bits", values="retained")
        p = p[[c for c in [8, 6, 5, 4, 3, 2] if c in p]]
        p.columns = [f"{c}b" for c in p.columns]
        awq = s[(s.method == "awq") & (s.task == t)].set_index("base_model").retained
        p["AWQ-4b"] = [awq.get(m) for m in p.index]
        p.index = [f"{m} ({MODELS[m][2]:.1f}B)" for m in p.index]
        p.index.name = "model"
        out.append(f"### Retained fraction — {t}\n\n" + md(p))
    # Table 3: mean retained by task x bits
    agg = sw.pivot_table(index="task", columns="bits", values="retained", aggfunc="mean")[[8, 6, 5, 4, 3, 2]].reindex(TASKS)
    agg.columns = [f"{c}b" for c in agg.columns]
    out.append("### Table B. Mean retained fraction across models (RTN g128)\n\n" + md(agg))
    # Critical bits
    cl = pd.read_csv(RES / "cliffs.csv")
    cb = cl.pivot(index="model", columns="task", values="critical_bits").reindex(columns=TASKS)
    cb.index = [f"{m} ({MODELS[m][2]:.1f}B)" for m in cb.index]; cb.index.name = "model"
    out.append("### Table C. Critical bits (highest precision with retained < 0.85; 1 = never)\n\n" + md(cb, "{:.0f}"))
    # Significance counts
    sig = pd.read_csv(RES / "paired_tests_sig_drops.csv", index_col=0)
    sig = sig[[c for c in ["8", "6", "5", "4", "3", "2"] if c in sig]]
    out.append("### Table D. Models with a significant drop vs bf16 (paired McNemar/Wilcoxon, BH q<0.05) / models tested\n\n" + md(sig))
    # Error taxonomy
    et = pd.read_csv(RES / "error_taxonomy_by_task_bits.csv")
    et = et[et.bits.isin([8, 5, 4, 3])].set_index(["task", "bits"])[["n_correct_bf16", "lost", "gained", "frac_lost", "share_fluent_errors"]]
    et.index = [f"{a} {b}b" for a, b in et.index]; et.index.name = "task/bits"
    out.append("### Table E. Answer flips relative to bf16 (pooled over models)\n\n" + md(et))
    # KL
    kv = pd.read_csv(RES / "kl_vs_accuracy.csv")
    kt = kv.pivot_table(index="task", columns="bits", values="kl", aggfunc="median")[[8, 6, 5, 4, 3, 2]].reindex(TASKS)
    kt.columns = [f"{c}b" for c in kt.columns]
    out.append("### Table F. Median task-specific KL(bf16‖quant) in nats/token across models\n\n" + md(kt, "{:.3f}"))
    wp = kv.drop_duplicates(["model", "bits"]).pivot(index="model", columns="bits", values="wiki_ppl_ratio")
    wp = wp[[c for c in [8, 6, 5, 4, 3, 2] if c in wp]]; wp.columns = [f"{c}b" for c in wp.columns]
    out.append("### Table G. WikiText-2 perplexity ratio (quantized / bf16)\n\n" + md(wp))
    rb = RES / "robustness_granularity.csv"
    if rb.exists():
        h = json.load(open(RES / "hypothesis_tests.json"))["robustness"]["mean_retained"]
        r = pd.DataFrame(h).T
        r.index.name = "granularity"
        out.append("### Table H. Mean retained fraction, group-128 vs per-channel RTN (Qwen2.5-1.5B, Llama-3.2-1B, all 6 tasks)\n\n" + md(r))
    (RES / "tables.md").write_text("\n\n".join(out) + "\n")
    print((RES / "tables.md").read_text()[:3000])


if __name__ == "__main__":
    main()
