"""
make_adapt_figures.py  -  adaptation figures for the IJRS manuscript (same style as make_figures.py)
  Figure_mechanism  candidate quality (best-of-three failure) vs selection cost, per configuration,
                    (a) centroid prompt, (b) extreme points (never used in training)
  Figure_qc         coverage vs failure among retained masks as the confidence threshold rises,
                    frozen SAM vs LoRA + head, (a) box, (b) centroid
  Figure_noise      shifted prompts: frozen SAM, LoRA + head (exact-prompt training), LoRA + head (shifted-prompt training)
  Figure_scale      re-drawn with a sequential blue scale for the input protocols (frees green for the adapter)

Run from the repository folder:  python make_adapt_figures.py          (CPU only, ~5-10 min; safe during GPU jobs)
Output: results_v4\\figures_paper\\  (EPS + 600-dpi TIFF + PNG)
"""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as mf                     # identical rcParams, sizes and save()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RES = mf.RES
NB, TAU = 2000, 0.5
RNG = np.random.default_rng(11)

# one colour + marker per configuration, used in every adaptation figure (Okabe-Ito)
CONF = {   # key: (folder(s), label, colour, marker)
    "frozen":  (["sam1_tile1024"], "Frozen SAM", "#000000", "o"),
    "head":    (["sam1lora-head_tile1024"], "IoU head only", "#56B4E9", "^"),
    "lora":    (["sam1lora-loraonly_tile1024"], "LoRA only", "#E69F00", "s"),
    "decoder": (["sam1lora-decoder_tile1024"], "Mask decoder only", "#CC79A7", "D"),
    "box":     (["sam1lora-v3_tile1024"], "Box-only LoRA", "#D55E00", "v"),
    "v4":      (["sam1lora-v4_tile1024", "sam1lora-v4s1_tile1024", "sam1lora-v4s2_tile1024"],
                "LoRA + head", "#009E73", "*"),
}
COLS = ["ann_id", "image_id", "strategy", "iou", "iou_oracle", "confidence"]
_cache = {}


def load(folder, base=RES, strategies=None):
    key = (base, folder)
    if key not in _cache:
        p = os.path.join(base, folder, "results.csv")
        if not os.path.exists(p):
            _cache[key] = None
        else:
            d = pd.read_csv(p, usecols=lambda c: c in COLS, low_memory=False)
            _cache[key] = d.drop_duplicates(["ann_id", "strategy"], keep="last")
    d = _cache[key]
    if d is None: return None
    return d if strategies is None else d[d.strategy.isin(strategies)]


def boot_rate(img, x):
    """image-cluster bootstrap 95% CI of a mean of per-instance values x (in %)."""
    g = pd.DataFrame({"x": x, "n": 1.0, "i": img}).groupby("i").sum().values
    idx = RNG.integers(0, len(g), size=(NB, len(g)))
    b = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


# ------------------------------------------------------------------------------------------ Figure_mechanism
def mech_point(folder, s):
    d = load(folder, strategies=[s])
    f = (d.iou < TAU).values.astype(float) * 100
    o = (d.iou_oracle < TAU).values.astype(float) * 100
    img = d.image_id.values
    return dict(x=o.mean(), y=(f - o).mean(), xci=boot_rate(img, o), yci=boot_rate(img, f - o))


# label offsets (data units) chosen so that no two labels collide
LABEL_POS = {
    "CP":  {"frozen": (2.0, 1.6, "left"), "head": (2.0, 1.6, "left"), "lora": (-1.6, -2.0, "right"),
            "box": (1.6, 1.9, "left"), "decoder": (-2.5, 2.4, "center"), "v4": (2.0, -1.6, "left")},
    "EXT": {"frozen": (-2.0, 1.9, "right"), "head": (-2.0, 1.4, "right"), "lora": (2.6, -1.9, "left"),
            "box": (-2.0, -2.4, "right"), "decoder": (2.2, 1.6, "left"), "v4": (2.2, -1.0, "left")},
}


def fig_mechanism():
    print("Figure_mechanism ...")
    fig, axes = plt.subplots(1, 2, figsize=(mf.FULL_W, 2.9))
    for ax, s, lab in zip(axes, ["CP", "EXT"], ["Centroid prompt (CP)", "Extreme points (EXT, not used in training)"]):
        pts = {}
        for k, (folders, name, col, mk) in CONF.items():
            P = [mech_point(f, s) for f in folders if load(f) is not None]
            if not P: print(f"  missing {folders}"); continue
            x, y = np.mean([p["x"] for p in P]), np.mean([p["y"] for p in P])
            if len(P) > 1:      # seeds: +-1 SD
                xe = np.std([p["x"] for p in P], ddof=1); ye = np.std([p["y"] for p in P], ddof=1)
                xerr, yerr = [[xe], [xe]], [[ye], [ye]]
            else:
                xerr = [[x - P[0]["xci"][0]], [P[0]["xci"][1] - x]]; yerr = [[y - P[0]["yci"][0]], [P[0]["yci"][1] - y]]
            pts[k] = (x, y)
            print(f"  {s} {name:<18} best-of-3 fail {x:5.1f}  selection cost {y:5.1f}  (n runs {len(P)})")
            ax.errorbar([x], [y], xerr=xerr, yerr=yerr, fmt=mk, color=col, mfc=col, mec="white" if k != "frozen" else col,
                        ms=9 if k == "v4" else 6, mew=0.6, elinewidth=0.8, capsize=1.5, zorder=4)
        lim, top = 75, 38
        for tot in range(10, 111, 10):              # constant overall failure: x + y = tot
            ax.plot([0, tot], [tot, 0], color="0.9", lw=0.5, zorder=0)
            if tot in (40, 60, 80, 100):
                ax.text(tot - top + 0.6, top - 0.6, f"{tot}%", fontsize=6, color="0.55", ha="left", va="top", zorder=0)
        if "frozen" in pts:
            for k, (x, y) in pts.items():
                if k == "frozen": continue
                ax.annotate("", xy=(x, y), xytext=pts["frozen"], zorder=2,
                            arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.65", shrinkA=5, shrinkB=6, mutation_scale=7))
        for k, (x, y) in pts.items():               # direct labels (identity not by colour alone)
            dx, dy, ha = LABEL_POS[s].get(k, (2.0, 1.4, "left"))
            ax.text(x + dx, y + dy, CONF[k][1], fontsize=7, ha=ha, va="center", color="0.15", zorder=5)
        ax.set_xlim(0, lim); ax.set_ylim(-0.8, top)
        ax.grid(False)
        ax.set_xlabel("No correct candidate, best-of-three failure (%)")
        ax.set_title(lab, loc="left", fontsize=9, pad=4)
    axes[0].set_ylabel("Selection cost (percentage points)")
    mf.panel_label(axes[0], "a"); mf.panel_label(axes[1], "b")
    fig.tight_layout(w_pad=2.0)
    mf.save(fig, "Figure_mechanism")


# ------------------------------------------------------------------------------------------ Figure_qc
def qc_curve(folder, s):
    d = load(folder, strategies=[s]).dropna(subset=["confidence"])
    c, fail = d.confidence.values, (d.iou < TAU).values
    o = np.argsort(-c, kind="stable"); c, fail = c[o], fail[o]
    n = np.arange(1, len(c) + 1)
    cov = 100 * n / len(c); fr = 100 * np.cumsum(fail) / n
    marks = {t: (100 * (c >= t).mean(), 100 * fail[c >= t].mean() if (c >= t).any() else np.nan) for t in (0.70, 0.80, 0.90)}
    return cov, fr, 100 * fail.mean(), marks


def fig_qc():
    print("Figure_qc ...")
    fig, axes = plt.subplots(1, 2, figsize=(mf.FULL_W, 2.6), sharey=True)
    for ax, s, lab in zip(axes, ["BB", "CP"], ["Box prompt (BB)", "Centroid prompt (CP)"]):
        for k in ["frozen", "v4"]:
            folder = CONF[k][0][0]; _, name, col, mk = CONF[k]
            if load(folder) is None: continue
            cov, fr, base, marks = qc_curve(folder, s)
            keep = np.arange(len(cov)) >= 499        # at least 500 retained instances: no noisy tail
            cov, fr = cov[keep], fr[keep]
            step = max(1, len(cov) // 2000)
            ax.plot(cov[::step], fr[::step], color=col, lw=1.3, ls="-" if k == "v4" else "--", label=name)
            ax.axhline(base, color=col, lw=0.6, ls=":")
            for t, (cv, f) in marks.items():
                ax.plot([cv], [f], mk if k == "v4" else "o", color=col, ms=6 if k == "v4" else 4, mec="white", mew=0.5, zorder=4)
                if k == "v4":                       # thresholds labelled once, on the adapted curve
                    ax.text(cv, f + 6.0, f"{t:.2f}", fontsize=6, color="0.3", ha="center" if cv > 4 else "right",
                            clip_on=False)
                    ax.plot([cv, cv], [f + 1.0, f + 5.4], color="0.6", lw=0.4, zorder=3)
                print(f"  {s} {name:<12} t={t:.2f}: keeps {cv:5.1f}%  failure among kept {f:5.1f}%")
        ax.set_xlim(100, 0)                        # filtering removes instances: read left to right
        ax.set_ylim(0, 62)
        ax.set_xlabel("Instances retained (%)")
        ax.set_title(lab, loc="left", fontsize=9, pad=4)
    axes[0].set_ylabel("Failure among retained (%)")
    axes[0].legend(frameon=False, loc="upper right")
    mf.panel_label(axes[0], "a"); mf.panel_label(axes[1], "b")
    fig.tight_layout(w_pad=1.5)
    mf.save(fig, "Figure_qc")


# ------------------------------------------------------------------------------------------ Figure_noise
def fig_noise():
    """(a) box, (b) centroid: failure as the prompt is shifted, frozen SAM / LoRA + head (exact-prompt training) /
    LoRA + head trained with shifted prompts. Error bars: image-cluster bootstrap 95% CI (shifted-prompt adapter:
    mean +- 1 SD over three seeds, from results_v4/analysis/jitter/seeds_noise.csv written by seeds_report.py)."""
    print("Figure_noise ...")
    NOISE = os.path.join(HERE, "results_v4_noise")
    models = [("frozen", "sam1_tile1024", "Frozen SAM", "#000000", "o", "--"),
              ("v4", "sam1lora-v4_tile1024", "LoRA + head, exact prompts", "#009E73", "*", "-"),
              ("jit", "sam1lora-v4jit_tile1024", "LoRA + head, shifted prompts", "#0072B2", "D", "-.")]
    levels = [0, 5, 10, 20]
    fig, axes = plt.subplots(1, 2, figsize=(mf.FULL_W, 2.6), sharey=True)
    for ax, (fam, pref, lab) in zip(axes, (("BB", "BBN", "Box prompt (BB)"), ("CP", "CPJ", "Centroid prompt (CP)"))):
        for k, folder, name, col, mk, ls in models:
            sp = os.path.join(HERE, "results_v4", "analysis", "jitter", "seeds_noise.csv")
            if k == "jit" and os.path.exists(sp):      # shifted-prompt adapter: mean +- 1 SD over seeds 42, 1, 2
                S = pd.read_csv(sp); S = S[(S.model == "v4jit") & (S.prompt == fam)]
                g = S.groupby("shift").fail.agg(["mean", "std", "count"]).reindex(levels)
                ys, sd = g["mean"].values, g["std"].values
                print(f"  {name:<30} {fam}: " + "  ".join(f"±{l}px {y:5.1f}±{s:.1f}%" for l, y, s in zip(levels, ys, sd))
                      + f"  (seeds per level: {list(g['count'].astype(int))})")
                ax.errorbar(range(len(levels)), ys, yerr=sd, color=col, ls=ls, marker=mk, ms=4.5, mfc=col, mec="white",
                            mew=0.5, lw=1.2, elinewidth=0.7, capsize=1.5, label=name)
                continue
            noisy, clean = load(folder, base=NOISE), load(folder, strategies=[fam])
            if noisy is None or clean is None:
                print(f"  no runs for {name} yet ({folder})"); continue
            ys, lo, hi = [], [], []
            for lv in levels:
                d = clean if lv == 0 else noisy[noisy.strategy == f"{pref}{lv}"]
                f = (d.iou < TAU).values.astype(float) * 100
                a_, b_ = boot_rate(d.image_id.values, f)
                ys.append(f.mean()); lo.append(f.mean() - a_); hi.append(b_ - f.mean())
            print(f"  {name:<30} {fam}: " + "  ".join(f"±{l}px {y:5.1f}%" for l, y in zip(levels, ys)))
            ax.errorbar(range(len(levels)), ys, yerr=[lo, hi], color=col, ls=ls, marker=mk, ms=7 if mk == "*" else 4.5,
                        mfc=col, mec="white", mew=0.5, lw=1.2, elinewidth=0.7, capsize=1.5, label=name)
        ax.set_xticks(range(len(levels))); ax.set_xticklabels(["0", "±5", "±10", "±20"])
        ax.set_xlabel("Prompt shift (px)")
        ax.set_ylim(0, 100)
        ax.set_title(lab, loc="left", fontsize=9, pad=4)
    axes[0].set_ylabel("Failure rate, IoU < 0.5 (%)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7)
    mf.panel_label(axes[0], "a"); mf.panel_label(axes[1], "b")
    fig.tight_layout(w_pad=1.5)
    mf.save(fig, "Figure_noise")


# ------------------------------------------------------------------------------------------ Figure_scale (recoloured)
def fig_scale_recoloured():
    print("Figure_scale (sequential blues for input protocols) ...")
    mf.STYLE.update({"whole": ("#9ECAE1", "o"), "legacy": ("#6BAED6", "s"),
                     "tile1024": ("#2171B5", "^"), "tile512": ("#08306B", "D")})
    mf.fig_scale()


if __name__ == "__main__":
    fig_mechanism()
    fig_qc()
    fig_noise()
    fig_scale_recoloured()
    print("done ->", mf.OUT)
