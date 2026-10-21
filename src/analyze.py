"""Aggregate scores + KL and test hypotheses H1-H4. Writes tables to results/ and figures to figures/.

Definitions (pre-registered in planning.md)
- retained(b) = score(b) / score(bf16) for a (model, task).
- cliff between adjacent tested precisions b_hi > b_lo: retained(b_hi) - retained(b_lo) > 0.15.
- critical bits = highest precision whose retained fraction < 0.85 (bits "1" = never failed).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models import MODELS, LEVELS  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
TASKS = ["arith", "factqa", "code", "instr", "reading", "creative"]
TNAME = {"arith": "Arithmetic (GSM8K)", "factqa": "Factual QA (TriviaQA)", "code": "Code (HumanEval)",
         "instr": "Instruction following (IFEval)", "reading": "Reading comp. (SQuAD F1)",
         "creative": "Creative writing (judge)"}
SWEEP = [m for m in MODELS if not m.endswith("-awq")]
rng = np.random.default_rng(0)


def load_items():
    df = pd.read_csv(RES / "item_scores.csv")
    jc = {}
    if (RES / "judge_cache.jsonl").exists():
        for l in open(RES / "judge_cache.jsonl"):
            r = json.loads(l)
            jc[r["key"]] = np.mean(list(r["scores"].values())) / 10 if r["scores"] else np.nan
    m = df.task == "creative"
    keys = df[m].model + "|" + df[m].method + "|" + df[m].bits.astype(str) + "|" + df[m].group.astype(str) + "|" + df[m].id
    df.loc[m, "score"] = keys.map(jc).values
    return df


def boot_ci(x, n=1000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    bs = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def summarize(df):
    rows = []
    for (mo, me, b, g, t), d in df.groupby(["model", "method", "bits", "group", "task"]):
        lo, hi = boot_ci(d.score)
        rows.append(dict(model=mo, method=me, bits=b, group=g, task=t, score=d.score.mean(), ci_lo=lo, ci_hi=hi,
                         n=d.score.notna().sum(), trunc_rate=d.truncated.mean(), distinct2=d.distinct2.mean()))
    s = pd.DataFrame(rows)
    base = s[(s.method == "rtn") & (s.bits == 16) & (s.group == 128)].set_index(["model", "task"]).score
    s["base_model"] = s.model.str.replace("-awq", "")
    s["retained"] = [r.score / base.get((r.base_model, r.task), np.nan) for r in s.itertuples()]
    s["family"] = s.base_model.map(lambda m: MODELS[m][1])
    s["params_b"] = s.base_model.map(lambda m: MODELS[m][2])
    return s


def paired_tests(df):
    """bf16 vs each level: McNemar (binary tasks) or Wilcoxon (continuous), BH-FDR across all tests."""
    rows = []
    d0 = df[(df.method == "rtn") & (df.group == 128)]
    for (mo, t), d in d0.groupby(["model", "task"]):
        piv = d.pivot_table(index="id", columns="bits", values="score")
        if 16 not in piv:
            continue
        for b in [x for x in LEVELS if x != 16 and x in piv]:
            pr = piv[[16, b]].dropna()
            if t in ("arith", "factqa", "code", "instr"):
                tab = pd.crosstab(pr[16] > 0.5, pr[b] > 0.5).reindex(index=[False, True], columns=[False, True], fill_value=0)
                p = mcnemar(tab.values, exact=True).pvalue
            else:
                diff = pr[16] - pr[b]
                p = 1.0 if (diff == 0).all() else stats.wilcoxon(pr[16], pr[b]).pvalue
            rows.append(dict(model=mo, task=t, bits=b, delta=pr[b].mean() - pr[16].mean(), p=p, n=len(pr)))
    r = pd.DataFrame(rows)
    r["p_bh"] = multipletests(r.p, method="fdr_bh")[1]
    r["sig"] = r.p_bh < 0.05
    return r


def cliffs(s):
    rows = []
    sw = s[(s.method == "rtn") & (s.group == 128)]
    for (mo, t), d in sw.groupby(["model", "task"]):
        full = d.set_index("bits").retained.reindex(LEVELS)
        d = full.dropna()  # some models were swept on a subset of levels (time constraints)
        if 16 not in d or len(d) < 3:
            continue
        lv = list(d.index)
        drops = {f"{hi}->{lo}": d[hi] - d[lo] for hi, lo in zip(lv[:-1], lv[1:])}
        crit = next((b for b in lv if d[b] < 0.85), 1)
        big = max(drops, key=drops.get)
        rows.append(dict(model=mo, task=t, family=MODELS[mo][1], params_b=MODELS[mo][2], critical_bits=crit,
                         largest_drop_step=big, largest_drop=drops[big], drop_8_to_4=full[8] - full[4],
                         cliff_8_to_4=(full[8] - full[4]) > 0.15, ret8=full[8], ret6=full[6], ret5=full[5], ret4=full[4],
                         ret3=full[3], ret2=full[2], levels_tested=",".join(map(str, lv)),
                         any_cliff_steps=",".join(k for k, v in drops.items() if v > 0.15)))
    return pd.DataFrame(rows)


def kl_analysis(s, cl):
    kls = [pd.read_csv(p) for p in sorted((RES / "kl").glob("*_g128.csv"))]
    if not kls:
        return None, {}
    kl = pd.concat(kls)
    wiki = kl[kl.task == "wikitext"][["model", "bits", "ppl"]].rename(columns={"ppl": "wiki_ppl"})
    w16 = wiki[wiki.bits == 16].set_index("model").wiki_ppl
    wiki["wiki_ppl_ratio"] = wiki.wiki_ppl / wiki.model.map(w16)
    m = s[(s.method == "rtn") & (s.group == 128)].merge(kl[kl.task != "wikitext"], on=["model", "bits", "task"]) \
        .merge(wiki, on=["model", "bits"])
    m = m[(m.bits != 16) & m.retained.notna()].copy()
    m["failed"] = m.retained < 0.85
    out = {"n_cells": len(m), "n_failed": int(m.failed.sum())}
    out["spearman_kl_retained"] = stats.spearmanr(m.kl, m.retained)._asdict()
    for col, sign in [("kl", 1), ("wiki_ppl_ratio", 1), ("top1_agree", -1)]:
        out[f"auc_{col}"] = roc_auc_score(m.failed, sign * m[col])
    # Optimal KL threshold (Youden J) with bootstrap CI.
    def youden(df):
        ths = np.unique(df.kl)
        best = max(ths, key=lambda th: ((df.kl >= th) & df.failed).sum() / max(1, df.failed.sum())
                   - ((df.kl >= th) & ~df.failed).sum() / max(1, (~df.failed).sum()))
        return best
    th = youden(m)
    bs = [youden(m.sample(len(m), replace=True, random_state=i)) for i in range(300)]
    out["youden_threshold"] = float(th); out["youden_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    for t0 in [0.3, th]:
        pred = m.kl >= t0
        out[f"at_{t0:.3f}"] = dict(sensitivity=float((pred & m.failed).sum() / max(1, m.failed.sum())),
                                   specificity=float((~pred & ~m.failed).sum() / max(1, (~m.failed).sum())))
    out["per_task_auc"] = {t: (roc_auc_score(d.failed, d.kl) if d.failed.nunique() == 2 else None)
                           for t, d in m.groupby("task")}
    out["kl_at_failure_by_task_median"] = m[m.failed].groupby("task").kl.median().to_dict()
    out["kl_at_healthy_by_task_median"] = m[~m.failed].groupby("task").kl.median().to_dict()
    # Lead-indicator test: for each (model, task) with a failure, KL at the last healthy level above it.
    lead = []
    for (mo, t), d in m.groupby(["model", "task"]):
        d = d.set_index("bits").reindex([b for b in LEVELS if b != 16]).dropna(subset=["kl"])
        crit = next((b for b in d.index if d.loc[b, "failed"] is True or d.loc[b, "failed"] == 1), None)
        if crit is None:
            continue
        i = list(d.index).index(crit)
        prev = d.index[i - 1] if i > 0 else None
        lead.append(dict(model=mo, task=t, critical_bits=crit, kl_at_crit=d.loc[crit, "kl"],
                         prev_bits=prev, kl_prev=d.loc[prev, "kl"] if prev else np.nan,
                         ret_prev=d.loc[prev, "retained"] if prev else np.nan))
    lead = pd.DataFrame(lead)
    out["lead_n"] = len(lead)
    out["lead_frac_prev_kl_ge_0.3"] = float((lead.kl_prev >= 0.3).mean())
    out["lead_frac_crit_kl_ge_0.3"] = float((lead.kl_at_crit >= 0.3).mean())
    out["lead_median_kl_prev"] = float(lead.kl_prev.median()); out["lead_median_kl_crit"] = float(lead.kl_at_crit.median())
    lead.to_csv(RES / "kl_lead_analysis.csv", index=False)
    m.to_csv(RES / "kl_vs_accuracy.csv", index=False)
    return m, out


def robustness(s, th):
    """Exp 4: per-channel (group=-1) vs group-128 RTN, plus out-of-sample check of the KL threshold."""
    pc = s[(s.method == "rtn") & (s.group == -1)]
    if pc.empty:
        return {}
    g = s[(s.method == "rtn") & (s.group == 128) & s.model.isin(pc.model.unique()) & s.bits.isin(pc.bits.unique())]
    tab = pd.concat([g.assign(gran="g128"), pc.assign(gran="per-channel")]).pivot_table(
        index=["model", "task"], columns=["gran", "bits"], values="retained").round(3)
    tab.to_csv(RES / "robustness_granularity.csv")
    out = {"mean_retained": pd.concat([g.assign(gran="g128"), pc.assign(gran="per-channel")])
           .groupby(["gran", "bits"]).retained.mean().round(3).unstack().to_dict()}
    kls = [pd.read_csv(p) for p in sorted((RES / "kl").glob("*_g-1.csv"))]
    if kls:
        kl = pd.concat(kls)
        m = pc.merge(kl[kl.task != "wikitext"], on=["model", "bits", "task", "group"]).dropna(subset=["retained"])
        m["failed"] = m.retained < 0.85
        out["oos_n"] = len(m); out["oos_failed"] = int(m.failed.sum())
        if m.failed.nunique() == 2:
            out["oos_auc_kl"] = roc_auc_score(m.failed, m.kl)
        for t0 in [0.3, th]:
            pred = m.kl >= t0
            out[f"oos_at_{t0:.3f}"] = dict(sensitivity=float((pred & m.failed).sum() / max(1, m.failed.sum())),
                                           specificity=float((~pred & ~m.failed).sum() / max(1, (~m.failed).sum())))
        m.to_csv(RES / "kl_vs_accuracy_perchannel.csv", index=False)
    return out


def plot_curves(s):
    sw = s[(s.method == "rtn") & (s.group == 128)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    xs = {b: i for i, b in enumerate(LEVELS)}
    cmap = {"Qwen2.5": plt.cm.Blues, "Llama-3": plt.cm.Oranges, "Phi-3": plt.cm.Greens}
    for ax, t in zip(axes.flat, TASKS):
        for mo in SWEEP:
            d = sw[(sw.model == mo) & (sw.task == t)].set_index("bits").reindex(LEVELS).dropna(subset=["retained"])
            if d.empty:
                continue
            fam = MODELS[mo][1]
            fam_models = [x for x in SWEEP if MODELS[x][1] == fam]
            shade = 0.45 + 0.5 * fam_models.index(mo) / max(1, len(fam_models) - 1)
            ax.plot([xs[b] for b in d.index], d.retained.values, "-o", ms=4, color=cmap[fam](shade),
                    label=f"{fam} {MODELS[mo][2]:.1f}B")
            awq = s[(s.model == mo + "-awq") & (s.task == t)]
            if len(awq):
                ax.plot(xs[4] + 0.12, awq.retained.iloc[0], "*", ms=11, color=cmap[fam](shade), mec="k", mew=0.5)
        ax.axhline(0.85, ls="--", c="gray", lw=1)
        ax.axvspan(xs[8], xs[4], color="gray", alpha=0.06)
        ax.set_title(TNAME[t]); ax.set_ylim(-0.05, 1.25); ax.grid(alpha=0.3)
        ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(["bf16" if b == 16 else f"{b}b" for b in LEVELS])
    axes[0, 0].set_ylabel("score / bf16 score"); axes[1, 0].set_ylabel("score / bf16 score")
    for ax in axes[1]:
        ax.set_xlabel("weight precision (RTN, group 128)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Retained task performance vs weight precision (dashed: 85% threshold; ★: real AWQ-INT4 checkpoint)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fig1_retained_vs_bits.png", dpi=150); plt.close(fig)


def plot_heatmap(cl):
    rows = [m for m in SWEEP if m in set(cl.model)]  # only models that were actually run
    piv = cl.pivot(index="model", columns="task", values="critical_bits").reindex(index=rows, columns=TASKS)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(piv.values, cmap="Reds", vmin=1, vmax=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, "none" if v == 1 else f"{int(v)}b", ha="center", va="center", fontsize=9,
                    color="white" if v >= 5 else "black")
    ax.set_xticks(range(len(TASKS))); ax.set_xticklabels([TNAME[t].split(" (")[0] for t in TASKS], rotation=25, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{MODELS[m][1]} {MODELS[m][2]:.1f}B" for m in rows])
    ax.set_title("Critical precision: highest bit width with <85% of bf16 score")
    fig.colorbar(im, ax=ax, label="bits"); fig.tight_layout()
    fig.savefig(FIG / "fig2_critical_bits_heatmap.png", dpi=150); plt.close(fig)


def plot_kl(m, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    mk = dict(zip(TASKS, "osD^vP"))
    for t in TASKS:
        d = m[m.task == t]
        ax.scatter(d.kl.clip(lower=1e-4), d.retained, s=18, marker=mk[t], label=TNAME[t].split(" (")[0], alpha=0.75)
    ax.set_xscale("log"); ax.axhline(0.85, ls="--", c="gray"); ax.axvline(0.3, ls=":", c="r", label="KL = 0.3 nats")
    ax.axvline(out["youden_threshold"], ls="-.", c="k", lw=1, label=f"best threshold {out['youden_threshold']:.2f}")
    ax.set_xlabel("task-specific KL(bf16 || quantized), nats/token"); ax.set_ylabel("retained score")
    ax.set_ylim(-0.05, 1.3); ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("KL vs retained accuracy (all model×task×level)")
    ax = axes[1]
    sw = m.groupby(["bits", "task"]).kl.median().unstack().reindex([b for b in LEVELS if b != 16])
    for t in TASKS:
        if t in sw:
            ax.plot(range(len(sw)), sw[t], "-o", label=TNAME[t].split(" (")[0])
    ax.set_xticks(range(len(sw))); ax.set_xticklabels([f"{b}b" for b in sw.index]); ax.set_yscale("log")
    ax.axhline(0.3, ls=":", c="r"); ax.set_ylabel("median KL across models (nats/token)"); ax.set_xlabel("weight precision")
    ax.set_title("Median task-specific KL vs precision"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig3_kl_vs_accuracy.png", dpi=150); plt.close(fig)


def main():
    df = load_items()
    s = summarize(df)
    s.to_csv(RES / "summary_scores.csv", index=False)
    tests = paired_tests(df); tests.to_csv(RES / "paired_tests.csv", index=False)
    cl = cliffs(s); cl.to_csv(RES / "cliffs.csv", index=False)
    sig = tests.groupby(["task", "bits"]).apply(lambda d: f"{int((d.sig & (d.delta < 0)).sum())}/{len(d)}")
    sig.unstack().to_csv(RES / "paired_tests_sig_drops.csv")  # models with a significant drop vs bf16 (BH q<0.05)
    plot_curves(s); plot_heatmap(cl)

    out = {}
    # H1/H2: task-level summary of retained fractions at 4 bits and 8->4 cliffs
    sw = s[(s.method == "rtn") & (s.group == 128)]
    out["retained_by_task_bits_median"] = sw.pivot_table(index="task", columns="bits", values="retained", aggfunc="median").round(3).to_dict()
    out["cliff_8_to_4_count_by_task"] = cl.groupby("task").cliff_8_to_4.sum().astype(int).to_dict()
    out["n_models"] = int(cl.model.nunique())
    out["sig_drop_counts"] = sig.unstack().to_dict()
    out["critical_bits_median_by_task"] = cl.groupby("task").critical_bits.median().to_dict()
    # Task sensitivity ranking: mean retained at 4 and 3 bits (Friedman test across models)
    for b in (4, 3):
        piv = sw[sw.bits == b].pivot(index="model", columns="task", values="retained").dropna(axis=1, how="all").dropna()
        out[f"friedman_retained_b{b}"] = dict(zip(["stat", "p"], map(float, stats.friedmanchisquare(*[piv[c] for c in piv]))))
        out[f"mean_retained_b{b}"] = piv.mean().round(3).to_dict()
    # H3: size vs critical bits (Spearman, all models and within Qwen family)
    out["spearman_size_critbits_all"] = stats.spearmanr(np.log(cl.params_b), cl.critical_bits)._asdict()
    q = cl[cl.family == "Qwen2.5"]
    out["spearman_size_critbits_qwen"] = stats.spearmanr(np.log(q.params_b), q.critical_bits)._asdict()
    out["critical_bits_by_model_mean"] = cl.groupby("model").critical_bits.mean().round(2).to_dict()
    # H4: KL
    m, kl_out = kl_analysis(s, cl)
    out["kl"] = kl_out
    if m is not None:
        plot_kl(m, kl_out)
        out["robustness"] = robustness(s, kl_out["youden_threshold"])
    json.dump(out, open(RES / "hypothesis_tests.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
