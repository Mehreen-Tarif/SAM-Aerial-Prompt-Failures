r"""
jitter_report.py -- results of the shifted-prompt (jitter) retraining, compared with the exact-prompt v4 model
(same seed 42) and frozen SAM, on the full iSAID validation set. CPU only.

  A. clean prompts (9 types): failure, best-of-three failure, selection cost, mIoU, calibration (Spearman, AUROC)
     A2. failure by object size (small / medium / large) for BB, CP, EXT
     A3. paired v4jit minus v4: dIoU [image-cluster 95% CI], dFailure, dSelection cost, image-level Wilcoxon, Holm
  B. shifted prompts (+-5/10/20 px boxes BBN*, centroids CPJ*): failure [95% CI] per model and level
     B2. by object size
     B3. paired v4jit minus v4 and v4jit minus frozen at every level (Holm over the 6 noisy prompts)
     B4. does the confidence score still flag failures under shifted prompts? (AUROC, Spearman)

Run from the repository folder:   python jitter_report.py
Output: results_v4\analysis\jitter\jitter_report.txt (+ csv files).
"""
import os
import numpy as np, pandas as pd
from scipy import stats

B = os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(B, "results_v4", "analysis", "jitter"); os.makedirs(OUT, exist_ok=True)
REP = open(os.path.join(OUT, "jitter_report.txt"), "w", encoding="utf-8")
def say(s=""): print(s); REP.write(str(s) + "\n"); REP.flush()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)

TAU, NB = 0.5, 2000
S9 = ["BB", "BPC", "CP", "CPN", "MP3", "MP5", "MP10", "EXT", "CPC"]
NOISY = ["BBN5", "BBN10", "BBN20", "CPJ5", "CPJ10", "CPJ20"]
STRATA = ["small", "medium", "large"]
MODELS = {"frozen": "sam1_tile1024", "v4": "sam1lora-v4_tile1024", "v4jit": "sam1lora-v4jit_tile1024"}
COLS = ["ann_id", "image_id", "category", "stratum", "strategy", "iou", "iou_oracle", "confidence"]


def load(path):
    if not os.path.exists(path): say(f"!! missing {path}"); return None
    D = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False).drop_duplicates(["ann_id", "strategy"], keep="last")
    piv = {v: D.pivot(index="ann_id", columns="strategy", values=v) for v in ["iou", "iou_oracle", "confidence"]}
    W = piv["iou"]
    M = D.drop_duplicates("ann_id").set_index("ann_id")[["image_id", "category", "stratum"]].reindex(W.index)
    return dict(W=W, O=piv["iou_oracle"].reindex(W.index), C=piv["confidence"].reindex(W.index), M=M)


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


def summary(R, strategies):
    rows = []
    for s in [s for s in strategies if s in R["W"]]:
        ok = R["W"][s].notna()
        u, o, c = R["W"][s][ok].values, R["O"][s][ok].values, R["C"][s][ok].values
        img = R["M"].image_id[ok].values
        lo, hi = boot_ci(img, (u < TAU) * 100.0)
        cm = ~np.isnan(c)
        rows.append(dict(strategy=s, n=int(ok.sum()), fail=100 * (u < TAU).mean(), ci_lo=lo, ci_hi=hi,
                         oracle_fail=100 * (o < TAU).mean(), sel_cost=100 * ((u < TAU).mean() - (o < TAU).mean()),
                         mIoU=u.mean(), spearman=stats.spearmanr(c[cm], u[cm]).correlation if cm.sum() > 10 else np.nan,
                         auroc=auroc(c, u >= TAU)))
    return pd.DataFrame(rows).set_index("strategy")


def by_size(R, strategies):
    rows = []
    for s in [s for s in strategies if s in R["W"]]:
        r = {"strategy": s}
        for z in STRATA:
            m = (R["M"].stratum == z) & R["W"][s].notna()
            r[z] = 100 * (R["W"][s][m] < TAU).mean(); r[f"n_{z}"] = int(m.sum())
        rows.append(r)
    return pd.DataFrame(rows).set_index("strategy")


def paired(R1, R2, strategies):
    rows = []
    for s in [s for s in strategies if s in R1["W"] and s in R2["W"]]:
        ids = R1["W"].index[R1["W"][s].notna()].intersection(R2["W"].index[R2["W"][s].notna()])
        img = R1["M"].image_id.loc[ids].values
        a, b = R1["W"].loc[ids, s].values, R2["W"].loc[ids, s].values
        d = a - b; lo, hi = boot_ci(img, d)
        f = (a < TAU).astype(float) - (b < TAU).astype(float); flo, fhi = boot_ci(img, f)
        sel1 = (a < TAU).astype(float) - (R1["O"].loc[ids, s].values < TAU)
        sel2 = (b < TAU).astype(float) - (R2["O"].loc[ids, s].values < TAU)
        per = pd.Series(d).groupby(img).mean()
        p = stats.wilcoxon(per).pvalue if (per != 0).any() else np.nan
        rows.append(dict(strategy=s, n=len(ids), dIoU=d.mean(), ci_lo=lo, ci_hi=hi, dFail_pp=100 * f.mean(),
                         fail_lo=100 * flo, fail_hi=100 * fhi, dSelCost_pp=100 * (sel1 - sel2).mean(),
                         images_better=int((per > 0).sum()), images_worse=int((per < 0).sum()), p=p))
    T = pd.DataFrame(rows)
    if len(T): T["p_holm"] = holm(T.p.fillna(1).values)
    return T


def show(title, T, name):
    say(f"\n=== {title}"); say(T.round(3).to_string()); T.to_csv(os.path.join(OUT, name))


# ------------------------------------------------------------ A. clean prompts
C = {k: load(os.path.join(B, "results_v4", f, "results.csv")) for k, f in MODELS.items()}
C = {k: v for k, v in C.items() if v is not None}
for k, R in C.items(): show(f"A. clean prompts, {k} ({MODELS[k]})", summary(R, S9), f"clean_summary_{k}.csv")
for k, R in C.items(): show(f"A2. clean failure (%) by object size, {k}", by_size(R, ["BB", "CP", "EXT"]), f"clean_size_{k}.csv")
if "v4jit" in C and "v4" in C:
    show("A3. clean, v4jit minus v4 (positive dIoU / negative dFail = shifted-prompt training better)",
         paired(C["v4jit"], C["v4"], S9).set_index("strategy"), "clean_paired_v4jit_vs_v4.csv")

# ------------------------------------------------------------ B. shifted prompts
N = {k: load(os.path.join(B, "results_v4_noise", f, "results.csv")) for k, f in MODELS.items()}
N = {k: v for k, v in N.items() if v is not None}
say("\n=== B. failure (%) [95% CI] with exact and shifted prompts")
rows = []
for k in N:
    for fam, clean_s, pref in (("box", "BB", "BBN"), ("centroid", "CP", "CPJ")):
        r = {"model": k, "prompt": fam}
        if k in C:
            t = summary(C[k], [clean_s]).loc[clean_s]; r["0px"] = f"{t.fail:.1f} [{t.ci_lo:.1f}, {t.ci_hi:.1f}]"
        t = summary(N[k], [f"{pref}{l}" for l in (5, 10, 20)])
        for l in (5, 10, 20):
            x = t.loc[f"{pref}{l}"]; r[f"+-{l}px"] = f"{x.fail:.1f} [{x.ci_lo:.1f}, {x.ci_hi:.1f}]"
        rows.append(r)
T = pd.DataFrame(rows); say(T.to_string(index=False)); T.to_csv(os.path.join(OUT, "noise_table.csv"), index=False)
for k, R in N.items(): show(f"B2. shifted-prompt failure (%) by object size, {k}", by_size(R, NOISY), f"noise_size_{k}.csv")
if "v4jit" in N and "v4" in N:
    show("B3. shifted prompts, v4jit minus v4", paired(N["v4jit"], N["v4"], NOISY).set_index("strategy"),
         "noise_paired_v4jit_vs_v4.csv")
if "v4jit" in N and "frozen" in N:
    show("B3. shifted prompts, v4jit minus frozen SAM", paired(N["v4jit"], N["frozen"], NOISY).set_index("strategy"),
         "noise_paired_v4jit_vs_frozen.csv")
for k, R in N.items():
    show(f"B4. shifted prompts, selection cost and calibration, {k}",
         summary(R, NOISY)[["fail", "oracle_fail", "sel_cost", "spearman", "auroc"]], f"noise_summary_{k}.csv")
say("\nDONE -> " + OUT)
