r"""
analysis_v4.py -- every table and figure for the paper, from the sam_eval_v4.py outputs.
READ-ONLY on results; writes into results_v4\analysis\ :
    report.txt                 everything below, human-readable
    T1_<run>.csv               descriptive table per run (mIoU, macro, F/Q_succ/Q_fail by stratum, oracle, empty)
    T2_<run>.csv               paired prompt comparisons per run (paired dIoU, cluster-bootstrap CI, rank-biserial, d, image Wilcoxon)
    F3_<run>.csv               failure-rate gaps by stratum + between-stratum differences (cluster CIs)
    fidelity_<run>.csv         common-success conditional IoU differences
    tau_<run>.csv              threshold sensitivity
    scale_failure.csv          failure rate by object size AT SAM's INPUT, every run
    protocol_2x2.csv           legacy vs native protocol x 25-largest vs full population
    models.csv / backbones.csv / noise.csv / lora.csv   when those runs exist
    fig_scale.png, fig_gap.png, fig_noise.png
Usage (from the repository folder; CPU only, safe while the GPU runs):
    python analysis_v4.py            [--nboot 2000] [--base <folder>]
Runs that are still in progress are analysed on the instances finished so far (flagged in the report).
"""
import os, sys, glob, time, argparse
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--nboot", type=int, default=2000)
A = ap.parse_args()
B, NB = A.base, A.nboot
OUT = os.path.join(B, "results_v4", "analysis"); os.makedirs(OUT, exist_ok=True)
REP = open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8")
def say(s=""):
    print(s); REP.write(str(s) + "\n"); REP.flush()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)

ORDER = ["small", "medium", "large"]
MAIN = ["BB", "BPC", "CPN", "CP", "MP3", "MP5", "MP10", "CPC", "EXT"]
PAIRS = [("BPC", "CP"), ("BPC", "CPC"), ("BPC", "BB"), ("BPC", "CPN"), ("BB", "CP"), ("BB", "MP10"),
         ("CP", "CPN"), ("BB", "CPN"), ("EXT", "CP"), ("EXT", "CPC")]
COLS = ["ann_id", "image_id", "category", "area_px", "stratum", "model", "mode", "strategy", "win_w", "win_h",
        "input_scale", "iou", "confidence", "pred_area", "iou_oracle", "was_exception"]
N_TOTAL = 116648

# ------------------------------------------------------------------ loading
def discover():
    runs = {}
    for p in sorted(glob.glob(os.path.join(B, "results_v4*", "*", "results.csv"))):
        group = os.path.basename(os.path.dirname(os.path.dirname(p)))
        name = os.path.basename(os.path.dirname(p)) + ("" if group == "results_v4" else f"@{group.replace('results_v4_', '')}")
        if group in ("results_v4_check", "results_v4_smoke"): continue
        runs[name] = p
    return runs

CACHE = {}
def load(name, path):
    if name in CACHE: return CACHE[name]
    D = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    D = D.drop_duplicates(["ann_id", "strategy"], keep="last")
    W = D.pivot(index="ann_id", columns="strategy", values="iou")
    W = W[W.notna().all(axis=1)]          # in-progress runs: keep only instances whose strategies are all written
    O = D.pivot(index="ann_id", columns="strategy", values="iou_oracle")
    E = D.pivot(index="ann_id", columns="strategy", values="pred_area")
    M = D.drop_duplicates("ann_id").set_index("ann_id")[["image_id", "category", "area_px", "stratum", "win_w", "win_h", "input_scale"]]
    M["eff"] = np.sqrt(M.area_px) * M.input_scale
    r = dict(W=W, O=O.reindex(W.index), E=E.reindex(W.index), M=M.reindex(W.index), n=len(W), complete=len(W) >= N_TOTAL)
    CACHE[name] = r
    return r

# ------------------------------------------------------------------ cluster bootstrap helpers
def img_sums(img, cols):
    g = pd.DataFrame({f"c{i}": np.asarray(c, float) for i, c in enumerate(cols)}); g["img"] = np.asarray(img)
    return g.groupby("img").sum().values

def boot_idx(n, seed=42):
    return np.random.default_rng(seed).integers(0, n, size=(NB, n))

def ci_ratio(img, num, den, seed=42):
    S = img_sums(img, [num, den]); idx = boot_idx(len(S), seed)
    b = S[idx, 0].sum(1) / np.where(S[idx, 1].sum(1) == 0, np.nan, S[idx, 1].sum(1))
    return np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)

def fmt_ci(est, lo, hi, p=4): return f"{est:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"

# ------------------------------------------------------------------ per-run analyses
def table1(R, strats, tau=.5):
    W, Mt, O, E = R["W"], R["M"], R["O"], R["E"]; rows = []
    for s in strats:
        x = W[s]; r = {"strategy": s, "mIoU": x.mean(),
                       "macro": x.groupby(Mt.category).mean().mean(), "F_all": 100 * (x < tau).mean()}
        for z in ORDER:
            xz = x[Mt.stratum == z]; F = (xz < tau).mean()
            r[f"F_{z[0]}"] = 100 * F; r[f"Qs_{z[0]}"] = xz[xz >= tau].mean(); r[f"Qf_{z[0]}"] = xz[xz < tau].mean()
        Fa = (x < tau).mean(); qs, qf = x[x >= tau].mean(), x[x < tau].mean()
        r["Qs_all"], r["Qf_all"] = qs, qf; r["decomp_check"] = (1 - Fa) * qs + Fa * qf - x.mean()
        r["oracleF_all"] = 100 * (O[s] < tau).mean(); r["selection_cost_pp"] = r["F_all"] - r["oracleF_all"]
        r["empty_pct"] = 100 * (E[s] == 0).mean()
        rows.append(r)
    return pd.DataFrame(rows).set_index("strategy")

def table2(R, pairs):
    W, img = R["W"], R["M"].image_id; rows = []
    for a, b in pairs:
        if a not in W or b not in W: continue
        d = (W[a] - W[b]).values; npos, nneg = (d > 0).sum(), (d < 0).sum()
        lo, hi = ci_ratio(img, d, np.ones_like(d))
        per = pd.Series(d).groupby(np.asarray(img)).mean()
        rows.append(dict(pair=f"{a}-{b}", dIoU=d.mean(), ci_lo=lo, ci_hi=hi, rank_biserial=(npos - nneg) / max(1, npos + nneg),
                         cohen_d=d.mean() / d.std(ddof=1), images_favour_first=int((per > 0).sum()), images_favour_second=int((per < 0).sum()),
                         image_wilcoxon_p=stats.wilcoxon(per).pvalue if (per != 0).any() else np.nan))
    return pd.DataFrame(rows).set_index("pair")

def fig3(R, pairs=(("CP", "BB"), ("CP", "BPC")), tau=.5):
    W, Mt = R["W"], R["M"]; rows = []
    for a, b in pairs:
        if a not in W or b not in W: continue
        cols = []
        for z in ORDER:
            m = (Mt.stratum == z).astype(float).values
            cols += [((W[a] < tau).astype(float).values - (W[b] < tau).astype(float).values) * m, m]
        S = img_sums(Mt.image_id, cols); idx = boot_idx(len(S)); gap = {}
        for k, z in enumerate(ORDER):
            num, den = S[idx, 2 * k].sum(1), S[idx, 2 * k + 1].sum(1)
            gap[z] = 100 * num / np.where(den == 0, np.nan, den)
            est = 100 * S[:, 2 * k].sum() / max(1, S[:, 2 * k + 1].sum())
            rows.append(dict(pair=f"{a}-{b}", item=z, gap_pp=est, ci_lo=np.nanpercentile(gap[z], 2.5), ci_hi=np.nanpercentile(gap[z], 97.5)))
        for z1, z2 in [("medium", "small"), ("large", "medium"), ("large", "small")]:
            dd = gap[z1] - gap[z2]; lo, hi = np.nanpercentile(dd, [2.5, 97.5])
            rows.append(dict(pair=f"{a}-{b}", item=f"{z1}-{z2}", gap_pp=np.nanmean(dd), ci_lo=lo, ci_hi=hi))
    T = pd.DataFrame(rows); T["excludes_0"] = (T.ci_lo > 0) | (T.ci_hi < 0); return T

def fidelity(R, pairs, tau=.5):
    W, img = R["W"], R["M"].image_id; rows = []
    for a, b in pairs:
        if a not in W or b not in W: continue
        both = ((W[a] >= tau) & (W[b] >= tau)).values
        d = np.where(both, (W[a] - W[b]).values, 0.0)
        lo, hi = ci_ratio(img, d, both.astype(float))
        rows.append(dict(pair=f"{a}-{b}", n_joint_success=int(both.sum()), dQ_joint=d.sum() / max(1, both.sum()), ci_lo=lo, ci_hi=hi,
                         Qa_own=W[a][W[a] >= tau].mean(), Qb_own=W[b][W[b] >= tau].mean()))
    return pd.DataFrame(rows).set_index("pair")

def tau_sens(R, strats):
    W = R["W"]; rows = []
    for t in [.3, .4, .5, .6, .75]:
        F = {s: 100 * (W[s] < t).mean() for s in strats}
        rows.append(dict(tau=t, **{f"F_{s}": v for s, v in F.items()}, ranking=" < ".join(sorted(F, key=F.get))))
    return pd.DataFrame(rows).set_index("tau")

def rank_stability(R, strats):
    W, img = R["W"], R["M"].image_id
    S = img_sums(img, [W[s].values for s in strats] + [np.ones(len(W))]); idx = boot_idx(len(S))
    m = S[idx, :len(strats)].sum(1) / S[idx, -1].sum(1)[:, None]
    full = np.argsort(np.argsort(-(S[:, :len(strats)].sum(0) / S[:, -1].sum())))
    r = np.argsort(np.argsort(-m, 1), 1)
    return 100 * (r == full).all(1).mean(), [strats[i] for i in np.argsort(full)]

# ------------------------------------------------------------------ main
t0 = time.time()
say(f"analysis_v4  {time.strftime('%Y-%m-%d %H:%M')}  base={B}  bootstrap B={NB} (image-level clusters)")
runs = discover()
main_runs = {k: v for k, v in runs.items() if "@" not in k}
say("runs found: " + ", ".join(f"{k} ({'complete' if load(k, v)['complete'] else str(load(k, v)['n']) + ' inst, IN PROGRESS'})" for k, v in runs.items()))

for name, path in main_runs.items():
    R = load(name, path); strats = [s for s in MAIN if s in R["W"]]
    say("\n" + "=" * 110 + f"\nRUN {name}   instances {R['n']:,}{'' if R['complete'] else '  (IN PROGRESS: partial)'}\n" + "=" * 110)
    T1 = table1(R, strats); T1.to_csv(os.path.join(OUT, f"T1_{name}.csv"))
    say("Descriptive table  (F = failure %, Qs = IoU | success, Qf = IoU | failure; mIoU = (1-F)Qs + F Qf, check column ~0)")
    say(T1[["mIoU", "macro", "F_all", "F_s", "F_m", "F_l", "Qs_s", "Qs_m", "Qs_l", "Qf_all", "oracleF_all", "selection_cost_pp", "empty_pct", "decomp_check"]].round(4).to_string())
    pct, order = rank_stability(R, strats)
    say(f"ranking (mean IoU): {' > '.join(order)}   reproduced in {pct:.1f}% of image resamples")
    T2 = table2(R, PAIRS); T2.to_csv(os.path.join(OUT, f"T2_{name}.csv"))
    say("\nPaired prompt comparisons  (cluster-bootstrap 95% CI over images; image-level Wilcoxon)"); say(T2.round(4).to_string())
    F3 = fig3(R); F3.to_csv(os.path.join(OUT, f"F3_{name}.csv"), index=False)
    say("\nFailure-rate gaps (pp) and between-stratum differences"); say(F3.round(2).to_string(index=False))
    FI = fidelity(R, PAIRS); FI.to_csv(os.path.join(OUT, f"fidelity_{name}.csv"))
    say("\nFidelity on the COMMON-success subset"); say(FI.round(4).to_string())
    TS = tau_sens(R, strats); TS.to_csv(os.path.join(OUT, f"tau_{name}.csv"))
    say("\nThreshold sensitivity (failure % at each tau)"); say(TS.round(2).to_string())

# ---- cross-run: same instances, equality check where tiling == whole image
say("\n" + "=" * 110 + "\nCROSS-RUN ANALYSES\n" + "=" * 110)
if "sam1_whole" in main_runs:
    Wh = load("sam1_whole", main_runs["sam1_whole"])
    for other in [k for k in main_runs if k.startswith("sam1_tile")]:
        T = load(other, main_runs[other]); common = Wh["W"].index.intersection(T["W"].index)
        same = common[(T["M"].win_w.reindex(common).values == Wh["M"].win_w.reindex(common).values) &
                      (T["M"].win_h.reindex(common).values == Wh["M"].win_h.reindex(common).values)]
        if len(same):
            d = (T["W"].loc[same, "BB"] - Wh["W"].loc[same, "BB"]).abs().max()
            say(f"sanity {other}: {len(same):,} instances whose window IS the whole image -> max |BB IoU diff| vs whole = {d:.2e} (must be 0)")

# ---- failure vs object size at SAM's input, every run
bins = [0, 2, 4, 8, 16, 32, 64, 128, 1e9]; rows = []
for name, path in main_runs.items():
    R = load(name, path); eb = pd.cut(R["M"].eff, bins, right=False)
    for s in [s for s in ["BB", "BPC", "CP", "EXT"] if s in R["W"]]:
        g = (R["W"][s] < .5).groupby(eb, observed=True)
        for b_, v in g.mean().items(): rows.append(dict(run=name, strategy=s, eff_bin=str(b_), n=int(g.size()[b_]), fail_pct=100 * v))
SC = pd.DataFrame(rows); SC.to_csv(os.path.join(OUT, "scale_failure.csv"), index=False)
say("\nFailure % by object side length AT SAM's INPUT (sqrt(area) x input scale)")
say(SC.pivot_table(index=["strategy", "eff_bin"], columns="run", values="fail_pct", sort=False).round(1).to_string())

# ---- complete, order-independent 2x2 (protocol x sample), on official val
if "sam1_whole" in main_runs and "sam1_legacy1024" in main_runs:
    N, L = load("sam1_whole", main_runs["sam1_whole"]), load("sam1_legacy1024", main_runs["sam1_legacy1024"])
    meta = N["M"]; top = meta.sort_values("area_px", ascending=False).groupby("category").head(25).index
    cells, rows = {}, []
    for pname, R in [("native", N), ("legacy", L)]:
        for sname, ids in [("top25", top), ("full", N["W"].index)]:
            ids = ids.intersection(R["W"].index); d = (R["W"].loc[ids, "BPC"] - R["W"].loc[ids, "BB"]).values
            lo, hi = ci_ratio(meta.image_id.reindex(ids), d, np.ones_like(d)); cells[(pname, sname)] = d.mean()
            rows.append(dict(protocol=pname, sample=sname, n=len(ids), BPC_minus_BB=d.mean(), ci_lo=lo, ci_hi=hi))
    P = pd.DataFrame(rows); P.to_csv(os.path.join(OUT, "protocol_2x2.csv"), index=False)
    prot = ((cells["legacy", "top25"] + cells["legacy", "full"]) - (cells["native", "top25"] + cells["native", "full"])) / 2
    samp = ((cells["native", "top25"] + cells["legacy", "top25"]) - (cells["native", "full"] + cells["legacy", "full"])) / 2
    inter = (cells["legacy", "top25"] - cells["legacy", "full"]) - (cells["native", "top25"] - cells["native", "full"])
    say("\n2x2 protocol design on official val (BPC - BB mean IoU)"); say(P.round(4).to_string(index=False))
    say(f"main effect of protocol (legacy - native) {prot:+.4f} | main effect of sample (top25 - full) {samp:+.4f} | interaction {inter:+.4f}")

# ---- SAM v1 vs SAM 2, same mode
rows = []
for mode in ["whole", "tile1024", "tile512"]:
    a, b = f"sam1_{mode}", f"sam2_{mode}"
    if a in main_runs and b in main_runs:
        R1, R2 = load(a, main_runs[a]), load(b, main_runs[b]); ids = R1["W"].index.intersection(R2["W"].index)
        for s in [s for s in MAIN if s in R1["W"] and s in R2["W"]]:
            d = (R2["W"].loc[ids, s] - R1["W"].loc[ids, s]).values; lo, hi = ci_ratio(R1["M"].image_id.reindex(ids), d, np.ones_like(d))
            f1, f2 = R1["W"].loc[ids, s] < .5, R2["W"].loc[ids, s] < .5
            rows.append(dict(mode=mode, strategy=s, n=len(ids), mIoU_sam1=R1["W"].loc[ids, s].mean(), mIoU_sam2=R2["W"].loc[ids, s].mean(),
                             d=d.mean(), ci_lo=lo, ci_hi=hi, F_sam1=100 * f1.mean(), F_sam2=100 * f2.mean()))
if rows:
    MD = pd.DataFrame(rows); MD.to_csv(os.path.join(OUT, "models.csv"), index=False)
    say("\nSAM 2 vs SAM v1 (identical prompts, same instances)"); say(MD.round(4).to_string(index=False))

# ---- backbones
bb_runs = {k: v for k, v in main_runs.items() if k.startswith("sam1") and k.endswith("_tile1024") and "lora" not in k}
if len(bb_runs) > 1:
    rows = []
    for k, v in bb_runs.items():
        R = load(k, v); T1 = table1(R, [s for s in MAIN if s in R["W"]])
        for s in T1.index: rows.append(dict(run=k, strategy=s, mIoU=T1.loc[s, "mIoU"], F_all=T1.loc[s, "F_all"], F_s=T1.loc[s, "F_s"], F_m=T1.loc[s, "F_m"], F_l=T1.loc[s, "F_l"]))
    BK = pd.DataFrame(rows); BK.to_csv(os.path.join(OUT, "backbones.csv"), index=False)
    say("\nBackbones (tile1024)"); say(BK.pivot(index="strategy", columns="run", values="mIoU").round(4).to_string())
    for k in bb_runs: say(f"  ranking {k}: {rank_stability(load(k, bb_runs[k]), [s for s in MAIN if s in load(k, bb_runs[k])['W']])[1]}")

# ---- noisy prompts
noise = {k: v for k, v in runs.items() if k.endswith("@noise")}
if noise:
    rows = []
    for k, v in noise.items():
        R = load(k, v); W, Mt = R["W"], R["M"]
        for fam, base, levels in [("box", "BB", ["BBN5", "BBN10", "BBN20"]), ("point", "CP", ["CPJ5", "CPJ10", "CPJ20"])]:
            for s, px in zip([base] + levels, [0, 5, 10, 20]):
                if s not in W: continue
                r = dict(run=k, family=fam, strategy=s, noise_px=px, mIoU=W[s].mean(), F_all=100 * (W[s] < .5).mean())
                for z in ORDER: r[f"F_{z[0]}"] = 100 * (W[s][Mt.stratum == z] < .5).mean()
                if s != base:
                    d = (W[s] - W[base]).values; r["d_vs_clean"] = d.mean(); r["ci_lo"], r["ci_hi"] = ci_ratio(Mt.image_id, d, np.ones_like(d))
                rows.append(r)
    NZ = pd.DataFrame(rows); NZ.to_csv(os.path.join(OUT, "noise.csv"), index=False)
    say("\nNoisy prompts (uniform +-px per box edge / centroid offset)"); say(NZ.round(4).to_string(index=False))

# ---- LoRA vs frozen base, same mode
rows = []
for k, v in main_runs.items():
    if "lora" not in k: continue
    mode = k.split("_", 1)[1]; base = f"sam1_{mode}"
    if base not in main_runs: continue
    L, Bs = load(k, v), load(base, main_runs[base]); ids = L["W"].index.intersection(Bs["W"].index)
    for s in [s for s in MAIN if s in L["W"]]:
        for z in [None] + ORDER:
            sel = ids if z is None else ids[Bs["M"].stratum.reindex(ids).values == z]
            d = (L["W"].loc[sel, s] - Bs["W"].loc[sel, s]).values
            df_ = (L["W"].loc[sel, s] < .5).astype(float).values - (Bs["W"].loc[sel, s] < .5).astype(float).values
            img = Bs["M"].image_id.reindex(sel)
            rows.append(dict(run=k, strategy=s, stratum=z or "all", n=len(sel), dIoU=d.mean(), dIoU_ci=ci_ratio(img, d, np.ones_like(d)),
                             dFail_pp=100 * df_.mean(), dFail_ci=tuple(100 * np.array(ci_ratio(img, df_, np.ones_like(df_))))))
if rows:
    LR = pd.DataFrame(rows); LR.to_csv(os.path.join(OUT, "lora.csv"), index=False)
    say("\nLoRA vs frozen SAM (same mode, paired)"); say(LR.to_string(index=False))

# ---- figures
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for (run, s), g in SC[SC.strategy.isin(["BB", "CP"])].groupby(["run", "strategy"]):
        ax.plot(range(len(g)), g.fail_pct, marker="o", label=f"{run} {s}", ls="-" if s == "BB" else "--")
    labs = SC.drop_duplicates("eff_bin").eff_bin.tolist(); ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=30, fontsize=7)
    ax.set_xlabel("object side length at SAM input (px)"); ax.set_ylabel("failure rate (%)"); ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scale.png"), dpi=200); plt.close(fig)
    F3s = [(k, pd.read_csv(os.path.join(OUT, f"F3_{k}.csv"))) for k in main_runs if os.path.exists(os.path.join(OUT, f"F3_{k}.csv"))]
    fig, ax = plt.subplots(figsize=(7, 4)); w = 0.8 / max(1, len(F3s))
    for i, (k, F) in enumerate(F3s):
        g = F[(F.pair == "CP-BB") & F["item"].isin(ORDER)]
        ax.bar(np.arange(3) + i * w, g.gap_pp, w, yerr=[g.gap_pp - g.ci_lo, g.ci_hi - g.gap_pp], capsize=3, label=k)
    ax.set_xticks(np.arange(3) + w * (len(F3s) - 1) / 2); ax.set_xticklabels(ORDER); ax.set_ylabel("CP - BB failure gap (pp)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_gap.png"), dpi=200); plt.close(fig)
    if noise:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for fam, g in NZ.groupby("family"): ax.plot(g.noise_px, g.F_all, marker="o", label=fam)
        ax.set_xlabel("prompt noise (+- px)"); ax.set_ylabel("failure rate (%)"); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_noise.png"), dpi=200); plt.close(fig)
    say("\nfigures written: fig_scale.png, fig_gap.png" + (", fig_noise.png" if noise else ""))
except Exception as e:
    say(f"(figures skipped: {e})")
say(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {OUT}")
