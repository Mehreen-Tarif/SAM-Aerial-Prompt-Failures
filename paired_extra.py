r"""
paired_extra.py -- statistics the adaptation section needs (CPU only; safe to run while a GPU job is training).

  A. iSAID val, paired comparisons between ADAPTED models on the same instances:
       v4 vs decoder-only (and vs every other adapted run found: v3, seeds, ablations)
       per prompt: dIoU [image-cluster 95% CI], dFailure, dSelection-cost [CI], images better/worse,
       image-level Wilcoxon, Holm over the 9 prompts.
  B. iSAID val, seed summary for v4 (mean +- SD over the seeds that exist): failure and selection cost.
  C. NWPU VHR-10: confidence calibration (Spearman, AUROC) and selection cost for frozen / v3 / v4,
       failure with and without harbours, and v4 vs frozen paired WITHOUT harbours (Holm).

Run from the repository folder:   python paired_extra.py
Output: results_v4\analysis\extra\paired_extra_report.txt (+ csv files)
"""
import os, glob
import numpy as np, pandas as pd
from scipy import stats

B = os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(B, "results_v4", "analysis", "extra"); os.makedirs(OUT, exist_ok=True)
REP = open(os.path.join(OUT, "paired_extra_report.txt"), "w", encoding="utf-8")
def say(s=""): print(s); REP.write(str(s) + "\n"); REP.flush()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)

TAU, NB = 0.5, 2000
S9 = ["BB", "BPC", "CP", "CPN", "MP3", "MP5", "MP10", "EXT", "CPC"]
COLS = ["ann_id", "image_id", "category", "stratum", "strategy", "iou", "iou_oracle", "confidence"]


def load(path):
    D = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False).drop_duplicates(["ann_id", "strategy"], keep="last")
    piv = {v: D.pivot(index="ann_id", columns="strategy", values=v) for v in ["iou", "iou_oracle", "confidence"] if v in D}
    W = piv["iou"]; W = W[W.notna().all(axis=1)]
    M = D.drop_duplicates("ann_id").set_index("ann_id")[["image_id", "category", "stratum"]].reindex(W.index)
    return dict(W=W, O=piv.get("iou_oracle", W * np.nan).reindex(W.index), C=piv.get("confidence", W * np.nan).reindex(W.index), M=M)


def boot_ci(img, x, seed=7):
    g = pd.DataFrame({"x": np.asarray(x, float), "n": 1.0, "i": np.asarray(img)}).groupby("i").sum().values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(NB, len(g)))
    b = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    ok = ~np.isnan(score); score, label = score[ok], label[ok]
    npos, nneg = label.sum(), (~label).sum()
    if npos == 0 or nneg == 0: return np.nan
    r = stats.rankdata(score)
    return (r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def paired(R1, R2, strategies, mask=None):
    ids = R1["W"].index.intersection(R2["W"].index)
    if mask is not None: ids = ids[mask(R1["M"].loc[ids])]
    img = R1["M"].image_id.loc[ids].values; rows = []
    for s in [s for s in strategies if s in R1["W"] and s in R2["W"]]:
        a, b = R1["W"].loc[ids, s].values, R2["W"].loc[ids, s].values
        d = a - b; lo, hi = boot_ci(img, d)
        f = (a < TAU).astype(float) - (b < TAU).astype(float)
        sel1 = (a < TAU).astype(float) - (R1["O"].loc[ids, s].values < TAU).astype(float)
        sel2 = (b < TAU).astype(float) - (R2["O"].loc[ids, s].values < TAU).astype(float)
        slo, shi = boot_ci(img, sel1 - sel2)
        per = pd.Series(d).groupby(img).mean()
        p = stats.wilcoxon(per).pvalue if (per != 0).any() else np.nan
        rows.append(dict(strategy=s, n=len(ids), dIoU=d.mean(), ci_lo=lo, ci_hi=hi, dFail_pp=100 * f.mean(),
                         dSelCost_pp=100 * (sel1 - sel2).mean(), sel_lo=100 * slo, sel_hi=100 * shi,
                         images_better=int((per > 0).sum()), images_worse=int((per < 0).sum()), p=p))
    T = pd.DataFrame(rows)
    if len(T): T["p_holm"] = holm(T.p.fillna(1).values)
    return T


def summary(R, strategies, mask=None):
    W, O, C, M = R["W"], R["O"], R["C"], R["M"]
    keep = np.ones(len(W), bool) if mask is None else np.asarray(mask(M))
    rows = []
    for s in [s for s in strategies if s in W]:
        u, o, c = W[s].values[keep], O[s].values[keep], C[s].values[keep]
        ok = ~np.isnan(c)
        rho = stats.spearmanr(c[ok], u[ok]).correlation if ok.sum() > 10 else np.nan
        rows.append(dict(strategy=s, n=int(keep.sum()), mIoU=np.nanmean(u), fail=100 * (u < TAU).mean(),
                         oracle_fail=100 * (o < TAU).mean(), sel_cost=100 * ((u < TAU).mean() - (o < TAU).mean()),
                         spearman=rho, auroc=auroc(c, u >= TAU), mean_conf=np.nanmean(c)))
    return pd.DataFrame(rows).set_index("strategy")


# ============================================================ A + B. iSAID
runs = {}
for p in sorted(glob.glob(os.path.join(B, "results_v4", "*", "results.csv"))):
    k = os.path.basename(os.path.dirname(p))
    if k == "sam1_tile1024" or k.startswith("sam1lora-"): runs[k] = p
say("iSAID runs found: " + ", ".join(runs))
R = {k: load(p) for k, p in runs.items()}
base = "sam1lora-v4_tile1024"
if base in R:
    for k in R:
        if k == base or k == "sam1_tile1024": continue
        say(f"\n=== A. {base} minus {k}  (iSAID val, paired; positive dIoU = v4 better; negative dFail/dSelCost = v4 better)")
        T = paired(R[base], R[k], S9); say(T.round(4).to_string(index=False))
        T.to_csv(os.path.join(OUT, f"paired_v4_vs_{k}.csv"), index=False)

seeds = [k for k in ["sam1lora-v4_tile1024", "sam1lora-v4s1_tile1024", "sam1lora-v4s2_tile1024"] if k in R]
say(f"\n=== B. v4 seed summary over {len(seeds)} run(s): {seeds}")
if seeds:
    S = pd.concat({k: summary(R[k], S9) for k in seeds})
    agg = S.groupby(level=1)[["fail", "sel_cost", "mIoU", "spearman", "auroc"]].agg(["mean", "std"]).reindex(S9)
    say(agg.round(3).to_string()); agg.to_csv(os.path.join(OUT, "v4_seeds.csv"))
say("\n=== iSAID per-run summary (failure, selection cost, calibration)")
for k in R:
    T = summary(R[k], S9); T.to_csv(os.path.join(OUT, f"summary_{k}.csv"))
    say(f"-- {k}"); say(T.round(3).to_string())

# ============================================================ C. NWPU
say("\n" + "=" * 100 + "\nC. NWPU VHR-10")
N = {}
for p in sorted(glob.glob(os.path.join(B, "results_ext_nwpu", "*", "results.csv"))):
    N[os.path.basename(os.path.dirname(p))] = load(p)
no_harb = lambda M: (M.category != "harbor").values
for k, Rk in N.items():
    for lab, m in [("all objects", None), ("without harbours", no_harb)]:
        T = summary(Rk, S9, m); T.to_csv(os.path.join(OUT, f"nwpu_summary_{k}_{lab.replace(' ', '_')}.csv"))
        say(f"-- {k}, {lab}"); say(T.round(3).to_string())
if "sam1lora-v4_tile1024" in N and "sam1_tile1024" in N:
    for lab, m in [("all objects", None), ("without harbours", no_harb)]:
        say(f"\n=== C. NWPU v4 minus frozen, {lab} (paired, Holm over 9 prompts)")
        T = paired(N["sam1lora-v4_tile1024"], N["sam1_tile1024"], S9, m); say(T.round(4).to_string(index=False))
        T.to_csv(os.path.join(OUT, f"nwpu_paired_v4_vs_frozen_{lab.replace(' ', '_')}.csv"), index=False)
say("\nDONE -> " + OUT)
