"""
audit_extra2.py -- READ-ONLY, CPU only (safe while the GPU trains). Robustness checks.

  H1 prompt_construction.csv  CP/MP prompts use the FIRST polygon of multi-part objects (inherited from v3).
                              Re-run the key comparisons on SINGLE-polygon objects only: do conclusions change?
  H2 size_vs_scale.csv        Failure by native object size x input scale (pooled whole/1024/512 runs), and a
                              PAIRED test on the SAME objects: 1024-px tiles (scale 1) vs 512-px tiles (scale 2)
                              and whole image (scale < 1) vs 1024-px tiles -> effect of scale with the object fixed.
  H3 quality_control.csv      Practical use of SAM's confidence: keep masks with confidence >= t;
                              share of objects kept (coverage) and failure among kept (risk).
  H4 windows.txt              How many objects needed their own crop instead of a grid tile, and their results.
  H5 lora_by_size.csv         Where LoRA's gains come from: BB/CP failure by size at the model input, frozen vs LoRA.

Run from the repository folder:
    python audit_extra2.py                       (iSAID val JSON path below; override with --val-json)
Output: results_v4\\analysis\\extra\\
"""
import os, json, argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--val-json", default=os.path.join(os.environ.get("ISAID_ROOT", r"D:\iSAID_Data"), "data", "iSAID", "Annotations", "iSAID_val_20190823_114742.json"))
ap.add_argument("--nboot", type=int, default=1000)
A = ap.parse_args()
RES = os.path.join(A.base, "results_v4"); OUT = os.path.join(RES, "analysis", "extra"); os.makedirs(OUT, exist_ok=True)
REP = open(os.path.join(OUT, "extra2_report.txt"), "w", encoding="utf-8")
def say(s=""): print(s); REP.write(str(s) + "\n"); REP.flush()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)

TAU, ORDER = 0.5, ["small", "medium", "large"]
STRATS = ["BB", "BPC", "CPN", "CP", "MP3", "MP5", "MP10", "CPC", "EXT"]
COLS = ["ann_id", "image_id", "category", "area_px", "stratum", "strategy", "iou", "confidence", "input_scale", "win_kind"]


def load(run):
    p = os.path.join(RES, run, "results.csv")
    if not os.path.exists(p): return None
    D = pd.read_csv(p, usecols=lambda c: c in COLS, low_memory=False).drop_duplicates(["ann_id", "strategy"], keep="last")
    W = D.pivot(index="ann_id", columns="strategy", values="iou")
    C = D.pivot(index="ann_id", columns="strategy", values="confidence").reindex(W.index)
    M = D.drop_duplicates("ann_id").set_index("ann_id")[["image_id", "category", "area_px", "stratum", "input_scale", "win_kind"]].reindex(W.index)
    print(f"  loaded {run}: {len(W):,}")
    return dict(W=W, C=C, M=M)


def boot_mean_ci(img, x, seed=42):
    g = pd.DataFrame({"x": np.asarray(x, float), "n": 1.0, "i": np.asarray(img)}).groupby("i").sum().values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(A.nboot, len(g)))
    b = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


RUNS = ["sam1_tile1024", "sam1_tile512", "sam1_whole", "sam1lora-v3_tile1024", "sam1lora-v4_tile1024"]
R = {r: load(r) for r in RUNS}; R = {k: v for k, v in R.items() if v is not None}
say("runs: " + ", ".join(R))

# ------------------------------------------------------------------ H1 prompt construction
say("\nH1. Prompt construction: single-polygon objects only (CP/MP use the first polygon; EXT uses the full mask)")
try:
    js = json.load(open(A.val_json, encoding="utf-8"))
    npoly = pd.Series({a["id"]: (len(a["segmentation"]) if isinstance(a["segmentation"], list) else 1) for a in js["annotations"]})
    say(f"  annotations: {len(npoly):,};  multi-polygon: {(npoly > 1).sum():,} ({100 * (npoly > 1).mean():.2f}%)")
    rows = []
    for run, X in R.items():
        if "tile512" in run or "whole" in run: continue
        W, M = X["W"], X["M"]; single = npoly.reindex(W.index).fillna(1) == 1
        for subset, sel in [("all", pd.Series(True, index=W.index)), ("single_polygon", single), ("multi_polygon", ~single)]:
            r = dict(run=run, subset=subset, n=int(sel.sum()))
            for s in STRATS:
                if s in W: r[f"{s}_mIoU"] = W.loc[sel, s].mean(); r[f"{s}_fail"] = 100 * (W.loc[sel, s] < TAU).mean()
            r["ranking"] = " > ".join(W.loc[sel, [s for s in STRATS if s in W]].mean().sort_values(ascending=False).index)
            rows.append(r)
    H1 = pd.DataFrame(rows); H1.to_csv(os.path.join(OUT, "prompt_construction.csv"), index=False)
    say(H1[["run", "subset", "n", "BB_fail", "CP_fail", "MP5_fail", "EXT_fail", "ranking"]].round(2).to_string(index=False))
except Exception as e:
    say(f"  !! H1 skipped: {e}  (pass --val-json with the iSAID val annotation path)")

# ------------------------------------------------------------------ H2 size vs scale
say("\nH2. Native object size vs input scale")
SB = [0, 4, 8, 16, 32, 64, np.inf]; SL = ["<4", "4-8", "8-16", "16-32", "32-64", ">=64"]
rows = []
for run in ["sam1_whole", "sam1_tile1024", "sam1_tile512"]:
    if run not in R: continue
    W, M = R[run]["W"], R[run]["M"]
    d = pd.DataFrame({"native": pd.cut(np.sqrt(M.area_px), SB, labels=SL, right=False), "scale": M.input_scale,
                      "BB": (W.BB < TAU).astype(float), "CP": (W.CP < TAU).astype(float)})
    d["scale_bin"] = pd.cut(d.scale, [0, 0.25, 0.5, 0.99, 1.01, 1.99, 2.01, 9], labels=["<=0.25", "0.25-0.5", "0.5-1", "1", "1-2", "2", ">2"])
    d["run"] = run; rows.append(d)
P = pd.concat(rows)
for s in ["BB", "CP"]:
    t = P.groupby(["native", "scale_bin"], observed=True)[s].agg(["mean", "size"])
    t = t[t["size"] >= 30]
    say(f"\n  {s} failure % by native side length (rows, px) x input scale (cols); cells with n<30 omitted")
    say((100 * t["mean"]).unstack().round(1).to_string())
pd.DataFrame(P.groupby(["native", "scale_bin"], observed=True)[["BB", "CP"]].agg(["mean", "size"])).to_csv(os.path.join(OUT, "size_vs_scale.csv"))

rows = []
for a, b, label in [("sam1_tile512", "sam1_tile1024", "2x upsampling (512 vs 1024 tiles), same objects"),
                    ("sam1_tile1024", "sam1_whole", "native vs downsampled (1024 tiles vs whole), same objects")]:
    if a not in R or b not in R: continue
    ids = R[a]["W"].index.intersection(R[b]["W"].index)
    Ma, Mb = R[a]["M"].loc[ids], R[b]["M"].loc[ids]
    keep = (Ma.input_scale - Mb.input_scale).abs() > 1e-3          # only objects whose scale actually differs
    ids = ids[keep.values]; nat = pd.cut(np.sqrt(Ma.loc[ids, "area_px"]), SB, labels=SL, right=False)
    for s in ["BB", "CP"]:
        for z in SL:
            sel = ids[(nat == z).values]
            if len(sel) < 30: continue
            dfail = (R[a]["W"].loc[sel, s] < TAU).astype(float).values - (R[b]["W"].loc[sel, s] < TAU).astype(float).values
            lo, hi = boot_mean_ci(Ma.loc[sel, "image_id"], dfail)
            rows.append(dict(comparison=label, strategy=s, native_side=z, n=len(sel),
                             fail_first=100 * (R[a]["W"].loc[sel, s] < TAU).mean(), fail_second=100 * (R[b]["W"].loc[sel, s] < TAU).mean(),
                             diff_pp=100 * dfail.mean(), ci_lo=100 * lo, ci_hi=100 * hi))
if rows:
    H2p = pd.DataFrame(rows); H2p.to_csv(os.path.join(OUT, "size_vs_scale_paired.csv"), index=False)
    say("\n  PAIRED, same objects (diff = first - second, failure percentage points, image-cluster 95% CI)")
    say(H2p.round(1).to_string(index=False))

# ------------------------------------------------------------------ H3 confidence-based quality control
say("\nH3. Quality control with SAM's confidence: keep masks with confidence >= t")
rows = []
for run, X in R.items():
    if "512" in run or "whole" in run: continue
    for s in ["BB", "CP", "MP5"]:
        if s not in X["W"]: continue
        c, f = X["C"][s], (X["W"][s] < TAU)
        for t in [0.0, 0.70, 0.80, 0.85, 0.90, 0.95]:
            k = c >= t
            rows.append(dict(run=run, strategy=s, threshold=t, coverage_pct=100 * k.mean(),
                             fail_among_kept_pct=100 * f[k].mean() if k.any() else np.nan,
                             failures_caught_pct=100 * (f & ~k).sum() / max(1, f.sum())))
H3 = pd.DataFrame(rows); H3.to_csv(os.path.join(OUT, "quality_control.csv"), index=False)
say(H3.round(1).to_string(index=False))

# ------------------------------------------------------------------ H4 windows
say("\nH4. Window type (grid tile vs own crop for objects that fit no tile)")
lines = []
for run in ["sam1_tile1024", "sam1_tile512"]:
    if run not in R: continue
    W, M = R[run]["W"], R[run]["M"]
    for kind, idx in M.groupby("win_kind").groups.items():
        lines.append(f"  {run:16s} {kind:6s} n={len(idx):7,d}  BB fail {100 * (W.loc[idx, 'BB'] < TAU).mean():5.1f}%  "
                     f"CP fail {100 * (W.loc[idx, 'CP'] < TAU).mean():5.1f}%")
say("\n".join(lines)); open(os.path.join(OUT, "windows.txt"), "w").write("\n".join(lines))

# ------------------------------------------------------------------ H5 LoRA gains by size
rows = []
base = R.get("sam1_tile1024")
for run, X in R.items():
    if "lora" not in run or base is None: continue
    ids = X["W"].index.intersection(base["W"].index)
    eff = pd.cut(np.sqrt(base["M"].loc[ids, "area_px"]) * base["M"].loc[ids, "input_scale"], SB, labels=SL, right=False)
    for s in ["BB", "CP", "CPN", "MP5", "EXT"]:
        if s not in X["W"]: continue
        for z in SL:
            sel = ids[(eff == z).values]
            if len(sel) < 30: continue
            rows.append(dict(run=run, strategy=s, side_at_input=z, n=len(sel),
                             fail_frozen=100 * (base["W"].loc[sel, s] < TAU).mean(), fail_lora=100 * (X["W"].loc[sel, s] < TAU).mean()))
if rows:
    H5 = pd.DataFrame(rows); H5["change_pp"] = H5.fail_lora - H5.fail_frozen
    H5.to_csv(os.path.join(OUT, "lora_by_size.csv"), index=False)
    say("\nH5. LoRA vs frozen by object side length at the model input (failure %)")
    say(H5.round(1).to_string(index=False))

say(f"\nDONE -> {OUT}")
