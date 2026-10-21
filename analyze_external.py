"""
analyze_external.py -- summarise an external-dataset evaluation (e.g. NWPU VHR-10) written by
    sam_eval_v4.py --out results_ext_nwpu ...
Same metrics and statistics as the iSAID analysis: failure (IoU < 0.5), mIoU, per size stratum (iSAID area
thresholds 32^2 / 96^2 px), image-cluster bootstrap 95% CIs, image-level Wilcoxon, per category,
and frozen vs LoRA paired comparisons for every LoRA run found.

Run from the repository folder:   python analyze_external.py --res results_ext_nwpu
"""
import os, glob, argparse
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--res", default="results_ext_nwpu")
ap.add_argument("--nboot", type=int, default=2000)
A = ap.parse_args()
RES = os.path.join(A.base, A.res); OUT = os.path.join(RES, "analysis"); os.makedirs(OUT, exist_ok=True)
REP = open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8")
def say(s=""): print(s); REP.write(str(s) + "\n"); REP.flush()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
TAU, ORDER = 0.5, ["small", "medium", "large"]
STRATS = ["BB", "BPC", "CPN", "CP", "MP3", "MP5", "MP10", "CPC", "EXT"]


def boot_ci(img, x, seed=42):
    g = pd.DataFrame({"x": np.asarray(x, float), "n": 1.0, "i": np.asarray(img)}).groupby("i").sum().values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(A.nboot, len(g)))
    b = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


runs = {}
for p in sorted(glob.glob(os.path.join(RES, "*", "results.csv"))):
    D = pd.read_csv(p, low_memory=False).drop_duplicates(["ann_id", "strategy"], keep="last")
    W = D.pivot(index="ann_id", columns="strategy", values="iou")
    M = D.drop_duplicates("ann_id").set_index("ann_id")[["image_id", "category", "area_px", "stratum"]].reindex(W.index)
    runs[os.path.basename(os.path.dirname(p))] = (W, M)
say(f"external dataset results in {RES}")
for k, (W, M) in runs.items():
    say(f"  {k}: {len(W):,} objects in {M.image_id.nunique()} images; strata " +
        ", ".join(f"{z} {100 * (M.stratum == z).mean():.1f}%" for z in ORDER))

for k, (W, M) in runs.items():
    say(f"\n=== {k}")
    rows = []
    for s in [s for s in STRATS if s in W]:
        r = dict(strategy=s, mIoU=W[s].mean(), fail=100 * (W[s] < TAU).mean())
        for z in ORDER:
            x = W.loc[M.stratum == z, s]; r[f"fail_{z}"] = 100 * (x < TAU).mean() if len(x) else np.nan
        rows.append(r)
    T = pd.DataFrame(rows).set_index("strategy"); say(T.round(3).to_string())
    T.to_csv(os.path.join(OUT, f"T1_{k}.csv"))
    if "BB" in W and "CP" in W:
        d = (W.BB - W.CP).values; lo, hi = boot_ci(M.image_id, d)
        per = pd.Series(d).groupby(M.image_id.values).mean()
        say(f"BB - CP mIoU {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  images favour BB {(per > 0).sum()} / CP {(per < 0).sum()}  "
            f"image Wilcoxon p = {stats.wilcoxon(per).pvalue:.2e}")
        g = ((W.CP < TAU).astype(float) - (W.BB < TAU).astype(float))
        for z in ORDER:
            sel = M.stratum == z
            if sel.sum() >= 30:
                lo, hi = boot_ci(M.image_id[sel], g[sel]); say(f"  CP-BB failure gap {z:6s} {100 * g[sel].mean():+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]  n={int(sel.sum())}")
    PC = pd.DataFrame([dict(category=c, n=len(i), BB_fail=100 * (W.loc[i, "BB"] < TAU).mean(), CP_fail=100 * (W.loc[i, "CP"] < TAU).mean())
                       for c, i in M.groupby("category").groups.items()]) if "BB" in W and "CP" in W else pd.DataFrame()
    if len(PC): say(PC.sort_values("n", ascending=False).round(1).to_string(index=False)); PC.to_csv(os.path.join(OUT, f"per_category_{k}.csv"), index=False)

base = next((k for k in runs if k.startswith("sam1_")), None)
for k in runs:
    if "lora" not in k or base is None: continue
    (WL, ML), (WB, MB) = runs[k], runs[base]; ids = WL.index.intersection(WB.index)
    say(f"\n=== {k} vs {base} (paired, same objects; transfer of the iSAID-trained adapter)")
    for s in [s for s in STRATS if s in WL and s in WB]:
        d = (WL.loc[ids, s] - WB.loc[ids, s]).values; lo, hi = boot_ci(MB.image_id.loc[ids], d)
        df_ = 100 * ((WL.loc[ids, s] < TAU).mean() - (WB.loc[ids, s] < TAU).mean())
        per = pd.Series(d).groupby(MB.image_id.loc[ids].values).mean()
        say(f"  {s:5s} dIoU {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  dFail {df_:+.1f} pp  images better {(per > 0).sum()} / worse {(per < 0).sum()}  "
            f"p = {stats.wilcoxon(per).pvalue if (per != 0).any() else float('nan'):.2e}")
say(f"\nDONE -> {OUT}")
