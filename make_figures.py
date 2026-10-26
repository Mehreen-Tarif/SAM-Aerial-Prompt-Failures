"""
make_figures.py  -  paper figures for the IJRS manuscript (Taylor & Francis artwork rules)
  Figure_scale  failure rate vs object side length at SAM's input (BB and CP panels)
  Figure_gap    CP - BB failure-rate gap by size stratum, with 95% bootstrap CIs
  Figure_noise  prompt-noise robustness

T&F artwork rules applied:
  * standard font (Arial/Helvetica), fonts embedded, consistent size
  * EPS (vector, recommended for line art) + TIFF 600 dpi (LZW) ; PNG only for viewing
  * no titles inside the image (captions go in the manuscript); panels labelled (a), (b)
  * readable when printed in greyscale: every series has its own marker / line style
  * no transparency, line weights >= 0.5 pt, supplied at intended print size

Run from the repository folder:  python make_figures.py
Output: results_v4/figures_paper/
CPU only - safe to run while the GPU is training.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v4")
ANA = os.path.join(RES, "analysis")
OUT = os.path.join(RES, "figures_paper")
os.makedirs(OUT, exist_ok=True)

# print sizes (inches). Full text width ~ 6.3 in, half width ~ 3.4 in
FULL_W, HALF_W = 6.3, 3.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "0.88", "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "legend.handlelength": 2.4, "legend.columnspacing": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType fonts
    "savefig.pad_inches": 0.02,
})

# Okabe-Ito colour-blind-safe palette; markers make every series distinct in greyscale
STYLE = {   # key: (colour, marker)
    "whole":    ("#D55E00", "o"),
    "legacy":   ("#E69F00", "s"),
    "tile1024": ("#0072B2", "^"),
    "tile512":  ("#009E73", "D"),
    "vit_l":    ("#56B4E9", "v"),
    "vit_b":    ("#CC79A7", "P"),
}


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.eps"), bbox_inches="tight")
    import io
    from PIL import Image                      # T&F: no transparency -> flatten RGBA to RGB, in memory
    buf = io.BytesIO()
    fig.savefig(buf, format="tiff", bbox_inches="tight", dpi=600, facecolor="white")
    buf.seek(0)
    rgb = Image.open(buf).convert("RGB")
    tif = os.path.join(OUT, f"{name}.tif")
    try:
        rgb.save(tif, format="TIFF", compression="tiff_lzw", dpi=(600, 600))
    except OSError:
        raise SystemExit(f"\n!! Windows could not overwrite {tif}\n"
                         f"   It is probably open in Photos / another viewer. Close it and run again.")
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.eps / .tif / .png")


def panel_label(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="right", va="bottom")


# --------------------------------------------------------------------------------------
# Figure_scale: failure vs effective object size at SAM's input
# --------------------------------------------------------------------------------------
EDGES = [0, 2, 4, 8, 16, 32, 64, 128, np.inf]
BIN_LABELS = ["<2", "2–4", "4–8", "8–16", "16–32", "32–64", "64–128", "≥128"]
MIN_N = 30          # bins with fewer instances are not drawn (too noisy)

RUNS_SCALE = [      # folder, legend label, style key, linestyle, hollow marker?
    ("sam1_whole",      "SAM, whole image",        "whole",    "-",  False),
    ("sam1_legacy1024", "SAM, square resize",    "legacy",   "-",  False),
    ("sam1_tile1024",   "SAM, 1024-px tiles",      "tile1024", "-",  False),
    ("sam1_tile512",    "SAM, 512-px tiles (2×)",  "tile512",  "-",  False),
    ("sam2_whole",      "SAM 2, whole image",      "whole",    "--", True),
    ("sam2_tile1024",   "SAM 2, 1024-px tiles",    "tile1024", "--", True),
]


def scale_table(folder):
    d = pd.read_csv(os.path.join(RES, folder, "results.csv"),
                    usecols=["strategy", "area_px", "input_scale", "iou"])
    d = d[d.strategy.isin(["BB", "CP"])].copy()
    eff = np.sqrt(d.area_px.astype(float)) * d.input_scale.astype(float)
    d["b"] = pd.cut(eff, EDGES, right=False, labels=False)
    d["fail"] = (d.iou < 0.5).astype(float)
    g = d.groupby(["strategy", "b"]).fail.agg(["size", "mean"])
    g.columns = ["n", "fail"]
    g["fail"] *= 100
    return g


def fig_scale():
    print("Figure_scale: recomputing size bins from raw results ...")
    tables = {f: scale_table(f) for f, *_ in RUNS_SCALE}

    print("  check (BB failure %, n>=30):")
    for f, *_ in RUNS_SCALE:
        s = tables[f].loc["BB"]
        row = ["{:>6}".format("-" if b not in s.index or s.loc[b, "n"] < MIN_N
                              else f"{s.loc[b, 'fail']:.1f}") for b in range(len(BIN_LABELS))]
        print(f"   {f:<16}" + "".join(row))

    x = np.arange(len(BIN_LABELS))
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), sharey=True)
    for ax, strat, lab_p, sub in zip(axes, ["BB", "CP"], ["(a)", "(b)"],
                                     ["Box prompt (BB)", "Centroid point (CP)"]):
        for f, lab, key, ls, hollow in RUNS_SCALE:
            col, mk = STYLE[key]
            y = np.full(len(x), np.nan)              # NaN = gap in the line, never a shift
            for b, row in tables[f].loc[strat].iterrows():
                if row.n >= MIN_N:
                    y[int(b)] = row.fail
            ax.plot(x, y, linestyle=ls, color=col, marker=mk, ms=3.6, lw=1.1,
                    mfc="white" if hollow else col, mec=col, mew=0.9, label=lab)
        ax.axvline(3.5, color="0.4", lw=0.7, ls=":")   # 16 px boundary
        ax.text(3.6, 99, "16 px\n(one ViT patch)", fontsize=6.5, color="0.3", va="top")
        ax.set_title(f"{lab_p} {sub}", loc="left", fontsize=9, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_xlabel("Object side length at model input (px)")
        ax.set_ylim(0, 102)
        ax.set_xlim(-0.4, len(x) - 0.6)
    axes[0].set_ylabel("Failure rate, IoU < 0.5 (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=1.5)
    save(fig, "Figure_scale")


# --------------------------------------------------------------------------------------
# Figure_gap: CP - BB failure gap per size stratum (dot + 95% CI, greyscale-safe)
# --------------------------------------------------------------------------------------
RUNS_GAP = [        # folder, label, style key, hollow?
    ("sam1_whole",          "SAM ViT-H, whole image",   "whole",    False),
    ("sam1_legacy1024",     "SAM ViT-H, square resize",        "legacy",   False),
    ("sam1_tile1024",       "SAM ViT-H, 1024-px tiles", "tile1024", False),
    ("sam1_tile512",        "SAM ViT-H, 512-px tiles",  "tile512",  False),
    ("sam1-vit_l_tile1024", "SAM ViT-L, 1024-px tiles", "vit_l",    False),
    ("sam1-vit_b_tile1024", "SAM ViT-B, 1024-px tiles", "vit_b",    False),
    ("sam2_whole",          "SAM 2, whole image",       "whole",    True),
    ("sam2_tile1024",       "SAM 2, 1024-px tiles",     "tile1024", True),
]
STRATA = ["small", "medium", "large"]


def fig_gap():
    print("Figure_gap: CP - BB gap ...")
    need = {"pair", "item", "gap_pp", "ci_lo", "ci_hi"}
    fig, ax = plt.subplots(figsize=(FULL_W, 2.5))
    k = len(RUNS_GAP); span = 0.72; step = span / (k - 1)
    for j, (f, lab, key, hollow) in enumerate(RUNS_GAP):
        col, mk = STYLE[key]
        d = pd.read_csv(os.path.join(ANA, f"F3_{f}.csv"))
        missing = need - set(d.columns)
        assert not missing, f"F3_{f}.csv lacks columns {missing}; columns are {list(d.columns)}"
        d = d[(d.pair == "CP-BB") & (d.item.isin(STRATA))].set_index("item").loc[STRATA]
        xs = np.arange(len(STRATA)) - span / 2 + step * j
        ax.errorbar(xs, d.gap_pp, yerr=[d.gap_pp - d.ci_lo, d.ci_hi - d.gap_pp],
                    fmt=mk, ms=4.2, color=col, mfc="white" if hollow else col, mec=col,
                    mew=1.0, elinewidth=0.9, capsize=1.8, capthick=0.8, label=lab)
    for xb in (0.5, 1.5):
        ax.axvline(xb, color="0.8", lw=0.5)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_xticks(range(len(STRATA)))
    ax.set_xticklabels(["Small", "Medium", "Large"])
    ax.set_xlim(-0.5, len(STRATA) - 0.5)
    ax.set_ylim(0, 50)
    ax.set_xlabel("Object size stratum")
    ax.set_ylabel("Failure-rate gap, CP − BB\n(percentage points)")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              handlelength=1.2, columnspacing=0.9, handletextpad=0.4, fontsize=7.5)
    fig.tight_layout()
    save(fig, "Figure_gap")


# --------------------------------------------------------------------------------------
# Figure_noise: prompt-noise robustness
# --------------------------------------------------------------------------------------
def fig_noise():
    print("Figure_noise: prompt noise ...")
    d = pd.read_csv(os.path.join(ANA, "noise.csv"))
    need = {"family", "noise_px", "F_all"}
    missing = need - set(d.columns)
    assert not missing, f"noise.csv lacks columns {missing}; columns are {list(d.columns)}"
    fig, ax = plt.subplots(figsize=(HALF_W, 2.5))
    for fam, lab, key, ls in (("box", "Box (BB), edges jittered", "tile1024", "-"),
                              ("point", "Centroid (CP), point jittered", "whole", "--")):
        col, mk = STYLE[key]
        s = d[d.family == fam].sort_values("noise_px")
        ax.plot(s.noise_px, s.F_all, ls, marker=mk, color=col, ms=4, lw=1.2, label=lab)
    clean_cp = d[(d.family == "point") & (d.noise_px == 0)].F_all.iloc[0]
    ax.axhline(clean_cp, color="0.4", lw=0.7, ls=":")
    ax.text(20, clean_cp - 1.5, f"clean centroid ({clean_cp:.1f}%)", fontsize=6.5,
            color="0.3", ha="right", va="top")
    ax.set_xticks([0, 5, 10, 20])
    ax.set_xlabel("Prompt perturbation (± px, uniform)")
    ax.set_ylabel("Failure rate, IoU < 0.5 (%)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "Figure_noise")


if __name__ == "__main__":
    import matplotlib.font_manager as fm
    used = fm.findfont(fm.FontProperties(family=plt.rcParams["font.sans-serif"]))
    print(f"font -> {os.path.basename(used)}   output -> {OUT}")
    fig_scale()
    fig_gap()
    fig_noise()
    print("done")
