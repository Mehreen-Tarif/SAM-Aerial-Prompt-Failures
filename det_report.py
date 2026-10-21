r"""
det_report.py -- SAM prompted with DETECTOR boxes vs ground-truth boxes, for frozen SAM, the exact-prompt adapter (v4)
and the shifted-prompt adapter (v4jit), iSAID validation set. CPU only.

  A. detected instances: failure with detector box vs ground-truth box (paired), per model and size stratum
  B. end to end over ALL instances: an instance the detector missed counts as a failure
  C. failure by the detector's box error (largest edge offset: <=2, 2-5, 5-10, >10 px), per model
  D. paired v4jit minus v4 and v4jit minus frozen with detector boxes (image-cluster CI, Wilcoxon)

Run from the repository folder:  python det_report.py      Output: results_det\det_report.txt
"""
import os, json
import numpy as np, pandas as pd
from scipy import stats

B = os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(B, "results_det"); os.makedirs(OUT, exist_ok=True)
REP = open(os.path.join(OUT, "det_report.txt"), "w", encoding="utf-8")
def say(s=""): print(s); REP.write(str(s) + "\n"); REP.flush()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
TAU, NB = 0.5, 2000
MODELS = {"frozen": "sam1_tile1024", "v4": "sam1lora-v4_tile1024", "v4jit": "sam1lora-v4jit_tile1024"}
COLS = ["ann_id", "image_id", "category", "stratum", "strategy", "iou", "iou_oracle", "confidence"]
STRATA = ["small", "medium", "large"]


def boot_ci(img, x, seed=7):
    g = pd.DataFrame({"x": np.asarray(x, float), "n": 1.0, "i": np.asarray(img)}).groupby("i").sum().values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(NB, len(g)))
    b = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def read(path, strat):
    d = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    return d[d.strategy == strat].drop_duplicates("ann_id", keep="last").set_index("ann_id")


det = json.load(open(os.path.join(B, "det", "det_boxes.json")))
det = pd.DataFrame.from_dict(det, orient="index"); det.index = det.index.astype(int)
R = {}
for k, f in MODELS.items():
    pd_, pg = os.path.join(OUT, f, "results.csv"), os.path.join(B, "results_v4", f, "results.csv")
    if not (os.path.exists(pd_) and os.path.exists(pg)): say(f"!! missing runs for {k}"); continue
    g = read(pg, "BB"); d = read(pd_, "DET")
    R[k] = g[["image_id", "stratum", "iou"]].rename(columns={"iou": "iou_gt"}).join(
        d[["iou", "iou_oracle", "confidence"]].rename(columns={"iou": "iou_det"}), how="left")
any_k = next(iter(R))
pop = R[any_k]; detected = pop.iou_det.notna()
say(f"population {len(pop):,} instances; detected and evaluated with the detector box {int(detected.sum()):,} "
    f"({100 * detected.mean():.1f}%)")
for z in STRATA:
    m = pop.stratum == z; say(f"  {z:<6} {int(m.sum()):>7,}  detected {100 * detected[m].mean():5.1f}%")

say("\n=== A. detected instances: failure (%) with ground-truth box vs detector box [95% CI]")
rows = []
for k, D in R.items():
    d = D[D.iou_det.notna()]
    for z in ["all"] + STRATA:
        e = d if z == "all" else d[d.stratum == z]
        fg, fd = (e.iou_gt < TAU) * 100.0, (e.iou_det < TAU) * 100.0
        lo, hi = boot_ci(e.image_id, fd - fg)
        rows.append(dict(model=k, stratum=z, n=len(e), fail_gt_box=fg.mean(), fail_det_box=fd.mean(),
                         diff_pp=(fd - fg).mean(), ci_lo=lo, ci_hi=hi, mIoU_gt=e.iou_gt.mean(), mIoU_det=e.iou_det.mean()))
T = pd.DataFrame(rows); say(T.round(2).to_string(index=False)); T.to_csv(os.path.join(OUT, "A_detected.csv"), index=False)

say("\n=== B. end to end over all instances (missed instance = failure): failure (%)")
rows = []
for k, D in R.items():
    for z in ["all"] + STRATA:
        e = D if z == "all" else D[D.stratum == z]
        f = ((e.iou_det.fillna(0)) < TAU) * 100.0; lo, hi = boot_ci(e.image_id, f)
        rows.append(dict(model=k, stratum=z, n=len(e), fail_end_to_end=f.mean(), ci_lo=lo, ci_hi=hi,
                         fail_gt_box=((e.iou_gt < TAU) * 100.0).mean()))
T = pd.DataFrame(rows); say(T.round(2).to_string(index=False)); T.to_csv(os.path.join(OUT, "B_end_to_end.csv"), index=False)

say("\n=== C. detected instances: failure (%) with the detector box by its largest edge offset")
err = det.copy()
try:
    from pycocotools.coco import COCO
    import sys; sys.path.insert(0, os.path.join(B, "scripts")); import sam_experiment_v3 as v3
    coco = COCO(v3.ANN_FILE)
    gb = {a["id"]: a["bbox"] for a in coco.loadAnns(list(det.index))}
    off = np.array([np.abs(np.array(det.loc[i, "box"]) - [gb[i][0], gb[i][1], gb[i][0] + gb[i][2], gb[i][1] + gb[i][3]]).max()
                    for i in det.index])
    err["max_off"] = off
    bins = pd.cut(err.max_off, [-0.01, 2, 5, 10, 1e9], labels=["<=2 px", "2-5 px", "5-10 px", ">10 px"])
    say("share of detected instances per bin: " + ", ".join(f"{b} {100 * v:.1f}%" for b, v in bins.value_counts(normalize=True).sort_index().items()))
    rows = []
    for k, D in R.items():
        d = D.join(bins.rename("bin"), how="inner")
        for b, e in d.groupby("bin", observed=True):
            rows.append(dict(model=k, max_edge_offset=b, n=len(e), fail_det_box=100 * (e.iou_det < TAU).mean(),
                             fail_gt_box=100 * (e.iou_gt < TAU).mean()))
    T = pd.DataFrame(rows); say(T.round(2).to_string(index=False)); T.to_csv(os.path.join(OUT, "C_by_offset.csv"), index=False)
    say(f"median largest edge offset {np.median(off):.1f} px; mean box IoU with ground truth {det.box_iou.mean():.3f}")
except Exception as e:
    say(f"(section C skipped: {e})")

say("\n=== D. paired comparisons with detector boxes (detected instances)")
for a_, b_ in [("v4jit", "v4"), ("v4jit", "frozen"), ("v4", "frozen")]:
    if a_ not in R or b_ not in R: continue
    ids = R[a_].index[R[a_].iou_det.notna()].intersection(R[b_].index[R[b_].iou_det.notna()])
    x, y = R[a_].loc[ids, "iou_det"].values, R[b_].loc[ids, "iou_det"].values; img = R[a_].loc[ids, "image_id"].values
    d = x - y; lo, hi = boot_ci(img, d); f = (x < TAU).astype(float) - (y < TAU); flo, fhi = boot_ci(img, f)
    per = pd.Series(d).groupby(img).mean()
    say(f"  {a_} minus {b_}: dIoU {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  dFailure {100 * f.mean():+.1f} pp "
        f"[{100 * flo:+.1f}, {100 * fhi:+.1f}]  images better/worse {(per > 0).sum()}/{(per < 0).sum()}  "
        f"Wilcoxon p {stats.wilcoxon(per).pvalue:.2g}")
say("\nDONE -> " + OUT)
