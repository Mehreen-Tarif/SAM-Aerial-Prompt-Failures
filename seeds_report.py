r"""
seeds_report.py -- variation over training seeds (42, 1, 2) for the exact-prompt (v4) and shifted-prompt (v4jit)
adapters, iSAID validation set. CPU only, about 10-15 min (reads one results file at a time).

  A. clean prompts: failure and selection cost per seed, mean +- SD over seeds (v4 and v4jit), by size for BB and CP
  B. paired, same seed: v4jit minus v4, failure difference [image-cluster 95% CI]
  C. shifted prompts: v4jit failure per seed and mean +- SD (v4 and frozen: one run each), small stratum too
  D. paired, shifted prompts: each v4jit seed minus v4 (seed 42) and minus frozen SAM

Run from the repository folder:   python seeds_report.py
Output: results_v4\analysis\jitter\seeds_report.txt and seeds_noise.csv
"""
import os
import numpy as np, pandas as pd

B = os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(B, "results_v4", "analysis", "jitter"); os.makedirs(OUT, exist_ok=True)
REP = open(os.path.join(OUT, "seeds_report.txt"), "w", encoding="utf-8")
def say(s=""): print(s); REP.write(str(s) + "\n"); REP.flush()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

TAU, NB = 0.5, 2000
S9 = ["BB", "BPC", "CP", "CPN", "MP3", "MP5", "MP10", "EXT", "CPC"]
NOISY = ["BBN5", "BBN10", "BBN20", "CPJ5", "CPJ10", "CPJ20"]
CLEAN = {("frozen", 0): "sam1_tile1024",
         ("v4", 42): "sam1lora-v4_tile1024", ("v4", 1): "sam1lora-v4s1_tile1024", ("v4", 2): "sam1lora-v4s2_tile1024",
         ("v4jit", 42): "sam1lora-v4jit_tile1024", ("v4jit", 1): "sam1lora-v4jits1_tile1024",
         ("v4jit", 2): "sam1lora-v4jits2_tile1024"}
NOISE = {("frozen", 0): "sam1_tile1024", ("v4", 42): "sam1lora-v4_tile1024",
         ("v4jit", 42): "sam1lora-v4jit_tile1024", ("v4jit", 1): "sam1lora-v4jits1_tile1024",
         ("v4jit", 2): "sam1lora-v4jits2_tile1024"}
COLS = ["ann_id", "image_id", "stratum", "strategy", "iou", "iou_oracle"]


def load(path):
    D = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False).drop_duplicates(["ann_id", "strategy"], keep="last")
    W = D.pivot(index="ann_id", columns="strategy", values="iou")
    O = D.pivot(index="ann_id", columns="strategy", values="iou_oracle").reindex(W.index)
    M = D.drop_duplicates("ann_id").set_index("ann_id")[["image_id", "stratum"]].reindex(W.index)
    return dict(W=W, O=O, M=M)


def boot_ci(img, x, seed=7):
    g = pd.DataFrame({"x": np.asarray(x, float), "n": 1.0, "i": np.asarray(img)}).groupby("i").sum().values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(NB, len(g)))
    b = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def stats_of(R, strategies):
    out = {}
    for s in [s for s in strategies if s in R["W"]]:
        ok = R["W"][s].notna(); u = R["W"][s][ok]; o = R["O"][s][ok]; z = R["M"].stratum[ok]
        lo, hi = boot_ci(R["M"].image_id[ok].values, (u.values < TAU) * 100.0)
        out[s] = dict(n=int(ok.sum()), fail=100 * (u < TAU).mean(), lo=lo, hi=hi,
                      sel=100 * ((u < TAU).mean() - (o < TAU).mean()),
                      **{f"fail_{k}": 100 * (u[z == k] < TAU).mean() for k in ("small", "medium", "large")})
    return out


def pdiff(R1, R2, s):
    ids = R1["W"].index[R1["W"][s].notna()].intersection(R2["W"].index[R2["W"][s].notna()])
    f = 100 * ((R1["W"].loc[ids, s].values < TAU).astype(float) - (R2["W"].loc[ids, s].values < TAU))
    lo, hi = boot_ci(R1["M"].image_id.loc[ids].values, f)
    return f"{f.mean():+5.1f} [{lo:+5.1f}, {hi:+5.1f}]"


def mean_sd(tab, model, s, key):
    v = [tab[(m, sd)][s][key] for (m, sd) in tab if m == model and s in tab[(m, sd)]]
    return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else np.nan, len(v))


# ------------------------------------------------------------------ A, B clean
say("A. clean prompts (iSAID val, 1024 tiles): failure % [95% CI] and selection cost (pp), per seed")
CS, keep = {}, {}
for key, f in CLEAN.items():
    p = os.path.join(B, "results_v4", f, "results.csv")
    if not os.path.exists(p): say(f"!! missing {p}"); continue
    R = load(p); CS[key] = stats_of(R, S9)
    if key[1] in (42, 1, 2): keep[key] = R          # kept for the paired comparison of the same seed
    say(f"\n  {key[0]} seed {key[1]} ({f})")
    for s, t in CS[key].items():
        say(f"    {s:<5} fail {t['fail']:5.1f} [{t['lo']:5.1f}, {t['hi']:5.1f}]  sel {t['sel']:5.1f}   "
            f"small {t['fail_small']:5.1f}  medium {t['fail_medium']:5.1f}  large {t['fail_large']:5.1f}")
    for sd in (42, 1, 2):                            # pair as soon as both models of a seed are loaded (limits memory)
        if True:
            if ("v4", sd) in keep and ("v4jit", sd) in keep:
                say(f"\nB. same seed {sd}, v4jit minus v4, failure difference (pp) [95% CI]:")
                for s in S9: say(f"    {s:<5} {pdiff(keep[('v4jit', sd)], keep[('v4', sd)], s)}")
                del keep[("v4", sd)], keep[("v4jit", sd)]
del keep

say("\nA (summary). mean +- SD over seeds 42, 1, 2: failure %, selection cost pp, small-object failure %")
say(f"  {'prompt':<6} {'frozen':>7} | {'v4 fail':>12} {'v4 sel':>11} {'v4 small':>12} | {'v4jit fail':>12} {'v4jit sel':>11} {'v4jit small':>12}")
for s in S9:
    fr = CS.get(("frozen", 0), {}).get(s, {}).get("fail", np.nan)
    cells = []
    for m in ("v4", "v4jit"):
        for k in ("fail", "sel", "fail_small"):
            a, b, n = mean_sd(CS, m, s, k); cells.append(f"{a:5.1f} +- {b:3.1f}")
    say(f"  {s:<6} {fr:7.1f} | " + " ".join(f"{c:>12}" for c in cells[:3]) + " | " + " ".join(f"{c:>12}" for c in cells[3:]))

# ------------------------------------------------------------------ C, D shifted prompts
say("\nC. shifted prompts: failure % [95% CI] per run")
NS, NR, rows = {}, {}, []
for key, f in NOISE.items():
    p = os.path.join(B, "results_v4_noise", f, "results.csv")
    if not os.path.exists(p): say(f"!! missing {p}"); continue
    R = load(p); NS[key] = stats_of(R, NOISY); NR[key] = R
    for fam, cs in (("BBN", "BB"), ("CPJ", "CP")):
        t0 = CS.get(key, {}).get(cs)
        if t0: rows.append(dict(model=key[0], seed=key[1], prompt=cs, shift=0, fail=t0["fail"], lo=t0["lo"], hi=t0["hi"], fail_small=t0["fail_small"]))
        for l in (5, 10, 20):
            t = NS[key][f"{fam}{l}"]
            rows.append(dict(model=key[0], seed=key[1], prompt=cs, shift=l, fail=t["fail"], lo=t["lo"], hi=t["hi"], fail_small=t["fail_small"]))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "seeds_noise.csv"), index=False)
for (m, sd), g in T.groupby(["model", "seed"]):
    say(f"  {m} seed {sd}: " + "  ".join(f"{r.prompt}{r.shift:>2}px {r.fail:5.1f}" for r in g.itertuples()))
say("\nC (summary). v4jit mean +- SD over 3 seeds (small objects in brackets); v4 and frozen one run")
for (pr, sh), g in T.groupby(["prompt", "shift"]):
    j = g[g.model == "v4jit"]; v = g[g.model == "v4"]; fz = g[g.model == "frozen"]
    say(f"  {pr} {sh:>2}px  frozen {fz.fail.mean():5.1f}  v4 {v.fail.mean():5.1f}  v4jit {j.fail.mean():5.1f} +- {j.fail.std(ddof=1):3.1f}"
        f"  (small: frozen {fz.fail_small.mean():5.1f}, v4 {v.fail_small.mean():5.1f}, v4jit {j.fail_small.mean():5.1f} +- {j.fail_small.std(ddof=1):3.1f})")

say("\nD. paired, shifted prompts: failure difference (pp) [95% CI]")
for sd in (42, 1, 2):
    if ("v4jit", sd) not in NR: continue
    for ref in (("v4", 42), ("frozen", 0)):
        if ref in NR:
            say(f"  v4jit seed {sd} minus {ref[0]}: " + "  ".join(f"{s} {pdiff(NR[('v4jit', sd)], NR[ref], s)}" for s in NOISY))
say("\nDONE -> " + OUT)
