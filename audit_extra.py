r"""
audit_extra.py -- READ-ONLY, CPU only (safe while the GPU trains). Additional analyses.

  A. per_category.csv      BB / CP / BPC mIoU and failure per iSAID category, every run, plus the
                           CP-BB failure gap per category with an image-level cluster-bootstrap 95% CI
  B. failure_anatomy.csv   WHY prompts fail, per stratum: selection error (a better candidate existed),
                           empty mask, under-segmentation (part of the object), leakage (spills over),
                           displaced (wrong place). Answers "why does CP fail on large objects?"
  C. holm.csv              paired prompt comparisons: image-level Wilcoxon p-values with Holm correction (per run)
  D. lora_paired.csv       LoRA vs frozen: image-level Wilcoxon + Holm, per strategy
  E. population.txt        stratum shares (shows micro mIoU is dominated by small objects)

Run from the repository folder:   python audit_extra.py
Output:                  results_v4\analysis\extra\
"""
import os, glob
import numpy as np, pandas as pd
from scipy import stats

B = os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(B, "results_v4")
OUT = os.path.join(RES, "analysis", "extra"); os.makedirs(OUT, exist_ok=True)
REP = open(os.path.join(OUT, "extra_report.txt"), "w", encoding="utf-8")
def say(s=""): print(s); REP.write(str(s) + "\n"); REP.flush()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)

ORDER = ["small", "medium", "large"]
STRATS = ["BB", "BPC", "CPN", "CP", "MP3", "MP5", "MP10", "CPC", "EXT"]
PAIRS = [("BPC", "CP"), ("BPC", "CPC"), ("BPC", "BB"), ("BPC", "CPN"), ("BB", "CP"), ("BB", "MP10"),
         ("CP", "CPN"), ("BB", "CPN"), ("EXT", "CP"), ("EXT", "CPC")]
COLS = ["ann_id", "image_id", "category", "area_px", "stratum", "strategy", "iou", "iou_oracle", "pred_area", "confidence"]
NB, TAU = 1000, 0.5
WANT = ["sam1_tile1024", "sam1lora-v3_tile1024", "sam1lora-v4_tile1024", "sam2_tile1024", "sam1_whole"]


def load(run):
    p = os.path.join(RES, run, "results.csv")
    if not os.path.exists(p): return None
    D = pd.read_csv(p, usecols=lambda c: c in COLS, low_memory=False).drop_duplicates(["ann_id", "strategy"], keep="last")
    W = D.pivot(index="ann_id", columns="strategy", values="iou")
    O = D.pivot(index="ann_id", columns="strategy", values="iou_oracle").reindex(W.index)
    E = D.pivot(index="ann_id", columns="strategy", values="pred_area").reindex(W.index)
    C = D.pivot(index="ann_id", columns="strategy", values="confidence").reindex(W.index)
    M = D.drop_duplicates("ann_id").set_index("ann_id")[["image_id", "category", "area_px", "stratum"]].reindex(W.index)
    print(f"  loaded {run}: {len(W):,} instances")
    return dict(W=W, O=O, E=E, M=M, C=C)


def boot_ci(img, num, den, seed=42):
    g = pd.DataFrame({"n": np.asarray(num, float), "d": np.asarray(den, float), "i": np.asarray(img)}).groupby("i").sum().values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(NB, len(g)))
    b = g[idx, 0].sum(1) / np.where(g[idx, 1].sum(1) == 0, np.nan, g[idx, 1].sum(1))
    return np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def img_wilcoxon(img, d):
    per = pd.Series(d).groupby(np.asarray(img)).mean()
    return (stats.wilcoxon(per).pvalue if (per != 0).any() else np.nan), int((per > 0).sum()), int((per < 0).sum())


say(f"audit_extra  base={B}")
runs = {r: load(r) for r in WANT}
runs = {k: v for k, v in runs.items() if v is not None}
say("runs: " + ", ".join(runs))

# ---------------------------------------------------------------- E. population
R0 = runs.get("sam1_tile1024") or next(iter(runs.values()))
Mt = R0["M"]
say("\nE. Population by stratum and category")
say((Mt.stratum.value_counts(normalize=True).reindex(ORDER) * 100).round(2).to_string())
say(Mt.groupby(["category", "stratum"]).size().unstack(fill_value=0).reindex(columns=ORDER).to_string())
open(os.path.join(OUT, "population.txt"), "w").write(Mt.groupby(["category", "stratum"]).size().unstack(fill_value=0).to_string())

# ---------------------------------------------------------------- A. per category
rows = []
for run, R in runs.items():
    W, M = R["W"], R["M"]
    for cat, idx in M.groupby("category").groups.items():
        r = dict(run=run, category=cat, n=len(idx), images=M.loc[idx, "image_id"].nunique())
        for s in ["BB", "BPC", "CP"]:
            if s in W: r[f"{s}_mIoU"] = W.loc[idx, s].mean(); r[f"{s}_fail"] = 100 * (W.loc[idx, s] < TAU).mean()
        if "BB" in W and "CP" in W:
            g = ((W.loc[idx, "CP"] < TAU).astype(float) - (W.loc[idx, "BB"] < TAU).astype(float)).values
            r["gap_CP_BB_pp"] = 100 * g.mean()
            lo, hi = boot_ci(M.loc[idx, "image_id"], g, np.ones_like(g)); r["gap_lo"], r["gap_hi"] = 100 * lo, 100 * hi
        rows.append(r)
PC = pd.DataFrame(rows); PC.to_csv(os.path.join(OUT, "per_category.csv"), index=False)
say("\nA. Per-category results (BB / CP), CP-BB failure gap with image-cluster 95% CI")
for run in runs:
    say(f"\n  {run}")
    say(PC[PC.run == run].drop(columns="run").set_index("category").sort_values("n", ascending=False).round(2).to_string())

# ---------------------------------------------------------------- B. failure anatomy
rows = []
for run, R in runs.items():
    W, O, E, M = R["W"], R["O"], R["E"], R["M"]
    for s in ["BB", "BPC", "CP", "CPN", "MP5", "EXT"]:
        if s not in W: continue
        for z in ORDER + ["all"]:
            sel = (M.stratum == z) if z != "all" else pd.Series(True, index=M.index)
            f = sel & (W[s] < TAU)
            nf = int(f.sum())
            if nf == 0: continue
            ratio = E.loc[f, s] / M.loc[f, "area_px"].clip(lower=1)
            selerr = O.loc[f, s] >= TAU
            empty = E.loc[f, s] == 0
            under = ~selerr & ~empty & (ratio < 0.5)
            leak = ~selerr & ~empty & (ratio > 2.0)
            other = ~selerr & ~empty & ~under & ~leak
            rows.append(dict(run=run, strategy=s, stratum=z, n=int(sel.sum()), fail_pct=100 * nf / max(1, int(sel.sum())),
                             of_failures_selection_error=100 * selerr.mean(), empty=100 * empty.mean(),
                             under_segmented_part=100 * under.mean(), leakage=100 * leak.mean(),
                             displaced_or_other=100 * other.mean(), median_pred_to_gt_area=float(ratio.median())))
FA = pd.DataFrame(rows); FA.to_csv(os.path.join(OUT, "failure_anatomy.csv"), index=False)
say("\nB. Failure anatomy (percent OF FAILURES; selection error = SAM's chosen mask failed but one of its 3 candidates succeeded)")
say(FA.round(1).to_string(index=False))

# ---------------------------------------------------------------- C. Holm on paired prompt comparisons
rows = []
for run, R in runs.items():
    W, M = R["W"], R["M"]; ps, tmp = [], []
    for a, b in PAIRS:
        if a not in W or b not in W: continue
        d = (W[a] - W[b]).values; p, fa, fb = img_wilcoxon(M.image_id, d)
        tmp.append(dict(run=run, pair=f"{a}-{b}", dIoU=d.mean(), images_favour_first=fa, images_favour_second=fb, p_raw=p)); ps.append(p)
    if tmp:
        adj = holm(np.nan_to_num(ps, nan=1.0))
        for t, q in zip(tmp, adj): t["p_holm"] = q; rows.append(t)
HT = pd.DataFrame(rows); HT.to_csv(os.path.join(OUT, "holm.csv"), index=False)
say("\nC. paired prompt comparisons, image-level Wilcoxon with Holm correction (10 comparisons per run)")
say(HT.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# ---------------------------------------------------------------- D. LoRA vs frozen, paired per image
rows = []
base = runs.get("sam1_tile1024")
for run, R in runs.items():
    if "lora" not in run or base is None: continue
    ids = R["W"].index.intersection(base["W"].index); img = base["M"].image_id.reindex(ids)
    ps, tmp = [], []
    for s in STRATS:
        if s not in R["W"] or s not in base["W"]: continue
        d = (R["W"].loc[ids, s] - base["W"].loc[ids, s]).values
        p, up, down = img_wilcoxon(img, d)
        lo, hi = boot_ci(img, d, np.ones_like(d))
        tmp.append(dict(run=run, strategy=s, dIoU=d.mean(), ci_lo=lo, ci_hi=hi, images_improved=up, images_worse=down, p_raw=p)); ps.append(p)
    adj = holm(np.nan_to_num(ps, nan=1.0))
    for t, q in zip(tmp, adj): t["p_holm"] = q; rows.append(t)
if rows:
    LP = pd.DataFrame(rows); LP.to_csv(os.path.join(OUT, "lora_paired.csv"), index=False)
    say("\nD. LoRA vs frozen SAM (1024-px tiles), image-level Wilcoxon + Holm over 9 strategies")
    say(LP.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# ---------------------------------------------------------------- F. confidence calibration
rows = []
for run, R in runs.items():
    W, C, M = R["W"], R["C"], R["M"]
    for s in STRATS:
        if s not in W: continue
        ok = C[s].notna()
        x, y = C.loc[ok, s].values, W.loc[ok, s].values
        succ = (y >= TAU).astype(int)
        auroc = stats.mannwhitneyu(x[succ == 1], x[succ == 0]).statistic / max(1, succ.sum() * (1 - succ).sum()) \
            if 0 < succ.sum() < len(succ) else np.nan
        rows.append(dict(run=run, strategy=s, spearman_conf_iou=stats.spearmanr(x, y).correlation,
                         auroc_conf_for_success=auroc, mean_conf=x.mean(), mean_iou=y.mean(),
                         overconfidence=x.mean() - y.mean()))
CF = pd.DataFrame(rows); CF.to_csv(os.path.join(OUT, "confidence.csv"), index=False)
say("\nF. Does SAM's confidence score track the true IoU? (higher Spearman / AUROC = better calibrated ranking)")
say(CF.round(3).to_string(index=False))

# ---------------------------------------------------------------- G. compute: image-encoder passes per protocol
import re
rows = []
for d in sorted(glob.glob(os.path.join(RES, "*", "log.txt"))):
    run = os.path.basename(os.path.dirname(d)); win = 0; mins = np.nan
    for line in open(d, encoding="utf-8", errors="ignore"):
        m = re.search(r"inst, (\d+) windows", line)
        if m: win += int(m.group(1))
        m = re.search(r"finished: .* in ([\d.]+) min", line)
        if m: mins = float(m.group(1))
    rows.append(dict(run=run, encoder_passes=win, wallclock_min_all_9_strategies=mins))
CP_ = pd.DataFrame(rows); CP_.to_csv(os.path.join(OUT, "compute.csv"), index=False)
say("\nG. Compute per protocol (encoder passes summed from log.txt; wall-clock includes all prompts; resumed runs undercount time)")
say(CP_.to_string(index=False))

say(f"\nDONE -> {OUT}")
