"""
make_qualitative.py  -  picture figures for the IJRS manuscript
  Figure_examples  real iSAID crops: ground truth | box prompt (BB) | centroid prompt (CP),
                   one row per size stratum (small / medium / large), SAM ViT-H, 1024-px tiles
  Figure_prompts   the nine prompt strategies drawn on one object (for the Methods section)

Examples are NOT hand-picked. Rule (state it in the caption):
  among instances where BB succeeds (IoU >= 0.75) and CP fails (IoU < 0.5) under the main protocol,
  take the one whose area is closest to the median area of its stratum; one image per row.

Needs: the same environment as sam_eval_v4.py (segment_anything, pycocotools, iSAID val, checkpoints)
       and make_figures.py in the same folder (shared fonts / colours).
Run from the repository folder:
    python make_qualitative.py                 # CPU (safe while LoRA training uses the GPU), ~3-6 min
    python make_qualitative.py --device cuda   # after training, faster
Output: results_v4/figures_paper/Figure_examples.* and Figure_prompts.*  (EPS + 600-dpi TIFF + PNG)
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as mf          # rcParams, save(), FULL_W, output folder -> identical style
import sam_eval_v4 as ev           # identical windows, prompts and model loading as the experiment
v3 = ev.v3
from pycocotools.coco import COCO

RUN = os.path.join(mf.RES, "sam1_tile1024", "results.csv")
STRATA = ["small", "medium", "large"]
BB_OK, CP_FAIL = 0.75, 0.5

# overlay colours (Okabe-Ito; distinguishable with red-green colour blindness)
C_TP, C_FP, C_FN = np.array([240, 228, 66]), np.array([213, 94, 0]), np.array([86, 180, 233])
ALPHA = 0.55
OUTLINE = [pe.withStroke(linewidth=2.2, foreground="black")]

NAMES = {"BB": "Box (BB)", "BPC": "Box + centroid (BPC)", "CP": "Centroid (CP)",
         "CPN": "Centroid + 3 negatives (CPN)", "MP3": "3 interior points (MP3)",
         "MP5": "5 interior points (MP5)", "MP10": "10 interior points (MP10)",
         "CPC": "Corners + centroid (CPC)", "EXT": "Extremes + centroid (EXT)"}


# ----------------------------------------------------------------------------- selection
def pick_examples():
    d = pd.read_csv(RUN, usecols=["ann_id", "image_id", "category", "area_px", "stratum", "strategy",
                                  "iou", "win_x", "win_y", "win_w", "win_h"])
    piv = d.pivot_table(index="ann_id", columns="strategy", values="iou")
    meta = d[d.strategy == "BB"].set_index("ann_id")
    meta = meta.join(piv[["BB", "CP"]])
    picks, used_imgs, used_cats = [], set(), set()
    for s in STRATA:
        all_s = meta[meta.stratum == s]
        bb_ok, cp_ok = all_s.BB >= 0.5, all_s.CP >= 0.5
        print(f"  {s:<6} outcomes (IoU>=0.5): both {(bb_ok & cp_ok).mean()*100:.1f}%  "
              f"BB only {(bb_ok & ~cp_ok).mean()*100:.1f}%  CP only {(~bb_ok & cp_ok).mean()*100:.1f}%  "
              f"neither {(~bb_ok & ~cp_ok).mean()*100:.1f}%")
        med = np.median(np.log(all_s.area_px))
        cand = all_s[(all_s.BB >= BB_OK) & (all_s.CP < CP_FAIL) & ~all_s.image_id.isin(used_imgs)
                     & ~all_s.category.isin(used_cats)].copy()   # one image and one category per row
        if cand.empty:
            raise SystemExit(f"no {s} instance with BB >= {BB_OK} and CP < {CP_FAIL}")
        cand["dist"] = (np.log(cand.area_px) - med).abs()
        row = cand.sort_values(["dist", "ann_id"]).iloc[0]
        used_imgs.add(row.image_id); used_cats.add(row.category)
        share = len(all_s[(all_s.BB >= BB_OK) & (all_s.CP < CP_FAIL)]) / len(all_s) * 100
        print(f"  {s:<6} ann {row.name} ({row.category}, {int(row.area_px)} px)  "
              f"BB {row.BB:.2f}  CP {row.CP:.2f}   [{share:.1f}% of {s} instances meet the rule]")
        picks.append((s, int(row.name), row))
    return picks


PROMPT_CATS = ["plane", "storage_tank", "Large_Vehicle", "ship", "Swimming_pool"]


def pick_prompt_example(coco, override=0):
    """A compact medium object (bbox aspect <= 1.6) so all nine prompts are distinguishable."""
    d = pd.read_csv(RUN, usecols=["ann_id", "category", "area_px", "stratum", "strategy",
                                  "win_x", "win_y", "win_w", "win_h"])
    d = d[(d.strategy == "BB") & (d.stratum == "medium")].set_index("ann_id")
    if override:
        return int(override), d.loc[int(override)]
    med = np.median(np.log(d.area_px))
    low = {c.lower(): i for i, c in enumerate(PROMPT_CATS)}
    d = d[d.category.str.lower().isin(low)].copy()
    asp = []
    for aid in d.index:
        _, _, w, h = coco.loadAnns([int(aid)])[0]["bbox"]
        asp.append(max(w, h) / max(1e-6, min(w, h)))
    d["asp"] = asp
    d = d[d.asp <= 1.6]
    d["pref"] = d.category.str.lower().map(low)
    d["dist"] = (np.log(d.area_px) - med).abs()
    aid = int(d.sort_values(["pref", "dist"]).index[0])
    r = d.loc[aid]
    print(f"  prompt figure: ann {aid} ({r.category}, {int(r.area_px)} px, aspect {r.asp:.2f})")
    return aid, r


# ----------------------------------------------------------------------------- helpers
def nice_cat(c):
    return c.replace("_", " ").strip().capitalize()      # 'Small_Vehicle' / 'ship' -> 'Small vehicle' / 'Ship'


def crop_box(bx, by, bw, bh, ww, wh, min_side=64, factor=2.6):
    """Square context crop around the object, in window coordinates, clipped to the window."""
    side = int(min(max(min_side, factor * max(bw, bh)), ww, wh))
    cx, cy = bx + bw / 2, by + bh / 2
    x0 = int(np.clip(round(cx - side / 2), 0, ww - side)); y0 = int(np.clip(round(cy - side / 2), 0, wh - side))
    return x0, y0, side


def overlay(img, gt, pred=None):
    out = img.astype(float).copy()
    if pred is None:
        layers = [(gt, C_TP)]
    else:
        layers = [(gt & pred, C_TP), (pred & ~gt, C_FP), (gt & ~pred, C_FN)]
    for m, c in layers:
        out[m] = (1 - ALPHA) * out[m] + ALPHA * c
    return out.astype(np.uint8)


def scale_bar(ax, side):
    nice = [5, 10, 20, 25, 50, 100, 200, 250, 500]
    L = min(nice, key=lambda v: abs(v - side / 4))
    x0, y0 = side * 0.06, side * 0.92
    ax.plot([x0, x0 + L], [y0, y0], color="white", lw=2.2, solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=3.6, foreground="black")])
    ax.text(x0 + L / 2, y0 - side * 0.03, f"{L} px", color="white", fontsize=6, ha="center",
            va="bottom", path_effects=OUTLINE)


def draw_prompt(ax, p, ox, oy, big=False):
    if p.get("box") is not None:
        x0, y0, x1, y1 = p["box"] - np.array([ox, oy, ox, oy])
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec="white", lw=1.1,
                               path_effects=[pe.withStroke(linewidth=2.2, foreground="black")]))
    if p.get("points") is not None:
        pts = np.asarray(p["points"], float) - [ox, oy]
        lab = np.asarray(p["labels"])
        pos, neg = pts[lab == 1], pts[lab == 0]
        ms = 5.5 if big else 7          # prompt figure: small enough that MP10's ten points separate
        # points are pixel indices (= pixel centres in imshow); boxes are COCO edge coordinates (-0.5)
        if len(pos):
            ax.plot(pos[:, 0], pos[:, 1], "*", ms=ms, mfc="white", mec="black", mew=0.8, ls="none")
        if len(neg):
            ax.plot(neg[:, 0], neg[:, 1], "X", ms=ms * 0.7, mfc="#D55E00", mec="black",
                    mew=0.7, ls="none")


def tag(ax, s):
    ax.text(0.04, 0.96, s, transform=ax.transAxes, fontsize=8, fontweight="bold", color="white",
            ha="left", va="top", path_effects=OUTLINE)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("0.3")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu (default, safe during training) or cuda")
    ap.add_argument("--prompt-ann", type=int, default=0, help="force a specific ann_id for Figure_prompts")
    args = ap.parse_args()

    print("selecting examples from", RUN)
    picks = pick_examples()

    img_dir, _ = v3.resolve_img_dir()
    coco = COCO(v3.ANN_FILE)
    predictor = ev.load_model("sam1", args.device, None, "vit_h")

    cells = []      # per row: dict with crop image, gt, preds, prompts, ious
    for s, aid, r in picks:
        a = coco.loadAnns([aid])[0]
        info = coco.loadImgs([a["image_id"]])[0]
        img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, info["file_name"])), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        wx, wy, ww, wh = int(r.win_x), int(r.win_y), int(r.win_w), int(r.win_h)
        win = np.ascontiguousarray(img[wy:wy + wh, wx:wx + ww])
        g, gx0, gy0 = ev.gt_local(a["segmentation"], a["bbox"], H, W)
        gt = np.zeros((wh, ww), bool); gt[gy0 - wy:gy0 - wy + g.shape[0], gx0 - wx:gx0 - wx + g.shape[1]] = g
        prompts = ev.build_all_prompts(a, H, W, g, gx0, gy0)
        predictor.set_image(win)
        out = {"stratum": s, "cat": r.category, "area": int(r.area_px), "prompts": {}, "pred": {}, "iou": {},
               "prompts_all": prompts, "wx": wx, "wy": wy}
        for st in ["BB", "CP"]:
            p = ev.shift_prompt(prompts[st], wx, wy)
            masks, scores, _ = predictor.predict(point_coords=p["points"], point_labels=p["labels"],
                                                 box=p["box"], multimask_output=True)
            m = np.asarray(masks)[int(np.argmax(scores))] > 0.5
            iou = (m & gt).sum() / max(1, (m | gt).sum())
            out["prompts"][st], out["pred"][st], out["iou"][st] = p, m, iou
            ref = getattr(r, st)
            flag = "" if abs(iou - ref) < 0.02 else "   <-- differs from results.csv (CPU vs GPU numerics?)"
            print(f"  ann {aid} {st}: IoU now {iou:.3f}, results.csv {ref:.3f}{flag}")
        bx, by, bw, bh = a["bbox"]
        out["crop"] = crop_box(bx - wx, by - wy, bw, bh, ww, wh)
        out["win"], out["gt"] = win, gt
        cells.append(out)

    # ---------------- Figure_examples
    fig, axes = plt.subplots(3, 3, figsize=(mf.FULL_W * 0.78, mf.FULL_W * 0.78 + 0.35))
    letters = iter("abcdefghi")
    for i, c in enumerate(cells):
        x0, y0, side = c["crop"]
        sl = (slice(y0, y0 + side), slice(x0, x0 + side))
        im, gt = c["win"][sl], c["gt"][sl]
        for j, col in enumerate(["GT", "BB", "CP"]):
            ax = axes[i, j]
            if col == "GT":
                ax.imshow(overlay(im, gt), interpolation="nearest")
                scale_bar(ax, side)
            else:
                ax.imshow(overlay(im, gt, c["pred"][col][sl]), interpolation="nearest")
                draw_prompt(ax, c["prompts"][col], x0, y0)
                ax.text(0.96, 0.04, f"IoU {c['iou'][col]:.2f}", transform=ax.transAxes, fontsize=7,
                        color="white", ha="right", va="bottom", path_effects=OUTLINE)
            ax.set_xlim(-0.5, side - 0.5); ax.set_ylim(side - 0.5, -0.5)
            tag(ax, f"({next(letters)})")
            clean(ax)
        axes[i, 0].set_ylabel(f"{c['stratum'].capitalize()}\n{nice_cat(c['cat'])}, {c['area']:,} px",
                              fontsize=8.5)
    for j, t in enumerate(["Ground truth", "Box prompt (BB)", "Centroid prompt (CP)"]):
        axes[0, j].set_title(t, fontsize=9, pad=4)
    handles = [Patch(fc=C_TP / 255, ec="0.3", lw=0.4, label="True positive / ground truth"),
               Patch(fc=C_FP / 255, ec="0.3", lw=0.4, label="False positive"),
               Patch(fc=C_FN / 255, ec="0.3", lw=0.4, label="False negative"),
               Line2D([], [], color="white", lw=1.1, label="Box prompt", path_effects=[pe.withStroke(linewidth=2.4, foreground="black")]),
               Line2D([], [], marker="*", ls="none", mfc="white", mec="black", ms=8, label="Point prompt")]
    fig.legend(handles=handles, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               fontsize=8)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    mf.save(fig, "Figure_examples")

    # ---------------- Figure_prompts (separate compact medium object; prompts only, no model)
    aid, r = pick_prompt_example(coco, args.prompt_ann)
    a = coco.loadAnns([aid])[0]
    info = coco.loadImgs([a["image_id"]])[0]
    img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, info["file_name"])), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    wx, wy, ww, wh = int(r.win_x), int(r.win_y), int(r.win_w), int(r.win_h)
    g, gx0, gy0 = ev.gt_local(a["segmentation"], a["bbox"], H, W)
    gtw = np.zeros((wh, ww), bool); gtw[gy0 - wy:gy0 - wy + g.shape[0], gx0 - wx:gx0 - wx + g.shape[1]] = g
    c = {"prompts_all": ev.build_all_prompts(a, H, W, g, gx0, gy0), "wx": wx, "wy": wy}
    bx, by, bw, bh = a["bbox"]
    x0, y0, side = crop_box(bx - wx, by - wy, bw, bh, ww, wh, factor=2.8)   # wider: CPN negatives visible
    sl = (slice(y0, y0 + side), slice(x0, x0 + side))
    im, gt = img[wy:wy + wh, wx:wx + ww][sl], gtw[sl]
    order = ["BB", "BPC", "CP", "CPN", "MP3", "MP5", "MP10", "CPC", "EXT"]
    fig, axes = plt.subplots(3, 3, figsize=(mf.FULL_W * 0.78, mf.FULL_W * 0.78 + 0.25))
    for k, (ax, st) in enumerate(zip(axes.flat, order)):
        ax.imshow(im, interpolation="nearest")
        ax.contour(gt.astype(float), levels=[0.5], colors=[C_TP / 255], linewidths=1.0)
        draw_prompt(ax, ev.shift_prompt(c["prompts_all"][st], c["wx"], c["wy"]), x0, y0, big=True)
        ax.set_xlim(-0.5, side - 0.5); ax.set_ylim(side - 0.5, -0.5)
        ax.set_title(NAMES[st], fontsize=8, pad=3)
        tag(ax, f"({'abcdefghi'[k]})")
        clean(ax)
    handles = [Line2D([], [], color=C_TP / 255, lw=1.2, label="Ground-truth outline"),
               Line2D([], [], color="white", lw=1.1, label="Box prompt", path_effects=[pe.withStroke(linewidth=2.4, foreground="black")]),
               Line2D([], [], marker="*", ls="none", mfc="white", mec="black", ms=8, label="Positive point"),
               Line2D([], [], marker="X", ls="none", mfc="#D55E00", mec="black", ms=6, label="Negative point")]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               fontsize=8)
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    mf.save(fig, "Figure_prompts")
    print("done")


if __name__ == "__main__":
    main()
