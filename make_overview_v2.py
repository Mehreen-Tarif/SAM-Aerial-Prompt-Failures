r"""
make_overview_v2.py  -  Figure 1 built from real iSAID data (replaces the box diagram of make_overview_figure.py)

  (a) input protocol and prompts: a validation image as SAM sees it when resized whole, with the 1024-px tile grid;
      the 1024-px tile at native resolution; the object with its box and centroid prompts
  (b) selection error: frozen SAM's three candidate masks for a centroid prompt - one is correct, another is returned;
      the mixed-prompt adapter (LoRA + IoU head) on the same prompt
  (c) no correct candidate: all three candidates of frozen SAM leak into the surroundings of a small vehicle;
      the adapter on the same prompt

Examples are chosen by a fixed rule from the existing per-instance results (printed when the script runs):
  (b) frozen SAM, centroid prompt, medium/large object, returned IoU < 0.5, best candidate IoU >= 0.75, adapter IoU >= 0.75,
      not a ship or harbour (shown in Figure 4), area 3,000-40,000 px, in an image whose longer side is >= 3,000 px
      (so that panel a shows the effect of resizing); the instance closest to the median log-area
  (c) frozen SAM, centroid prompt, small vehicle, every candidate > 2x the object area, adapter IoU >= 0.7;
      preferably in the same 1024-px tile as (b), otherwise the instance closest to the median log-area
Masks are recomputed (CPU by default, safe while the GPU trains) and compared with results.csv.

Run from the repository folder:   python make_overview_v2.py            (~5-10 min on CPU)
Output: results_v4\figures_paper\Figure_overview.(eps|tif|png)
"""
import os, sys, argparse, shutil, faulthandler
faulthandler.enable(file=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'overview_crash.txt'), 'w'), all_threads=True)
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Patch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as mf
import make_qualitative as mq
import sam_eval_v4 as ev
from pycocotools.coco import COCO
v3 = ev.v3

FROZEN = os.path.join(mf.RES, "sam1_tile1024", "results.csv")
ADAPT = os.path.join(mf.RES, "sam1lora-v4_tile1024", "results.csv")
LORA = os.path.join(HERE, "lora_v4", "tile1024", "best.pt")
COLS = ["ann_id", "image_id", "category", "area_px", "stratum", "strategy", "iou", "iou_oracle", "chosen",
        "iou_m0", "iou_m1", "iou_m2", "score_m0", "score_m1", "score_m2", "area_m0", "area_m1", "area_m2",
        "win_x", "win_y", "win_w", "win_h"]
BAND = {"a": "#EAF2FA", "b": "#F4F4F4", "c": "#F4F4F4"}
ORANGE, GREEN = "#E69F00", "#009E73"
OUT = [pe.withStroke(linewidth=2.0, foreground="black")]


def load_cp(path):
    parts = []
    for ch in pd.read_csv(path, usecols=lambda c: c in COLS, chunksize=400000, low_memory=False):
        parts.append(ch[ch.strategy == "CP"])
    return pd.concat(parts).drop_duplicates("ann_id", keep="last").set_index("ann_id")


def pick(coco):
    F, V = load_cp(FROZEN), load_cp(ADAPT)
    longside = {i["id"]: max(i["width"], i["height"]) for i in coco.loadImgs(coco.getImgIds())}
    F = F.join(V[["iou"]].rename(columns={"iou": "iou_ad"}), how="inner")
    cat = F.category.str.lower()
    s = F[(F.stratum != "small") & (F.iou < 0.5) & (F.iou_oracle >= 0.75) & (F.iou_ad >= 0.75)
          & ~cat.isin(["ship", "harbor"]) & F.area_px.between(3000, 40000)
          & (F.image_id.map(longside) >= 3000)].copy()
    if s.empty: raise SystemExit("no instance meets the rule for panel (b)")
    s["d"] = (np.log(s.area_px) - np.median(np.log(s.area_px))).abs()
    b = s.sort_values(["d"]).iloc[0]
    big = np.vstack([F[f"area_m{j}"] for j in range(3)]).T > 2 * F.area_px.values[:, None]
    l = F[(F.stratum == "small") & (cat == "small_vehicle") & big.all(1) & (F.iou_oracle < 0.5) & (F.iou_ad >= 0.7)].copy()
    if l.empty: raise SystemExit("no instance meets the rule for panel (c)")
    same = l[(l.image_id == b.image_id) & (l.win_x == b.win_x) & (l.win_y == b.win_y)]
    pool = same if len(same) else l
    pool = pool.assign(d=(np.log(pool.area_px) - np.median(np.log(l.area_px))).abs())
    c = pool.sort_values("d").iloc[0]
    print(f"  (b) ann {b.name}: {b.category}, {int(b.area_px)} px, frozen CP IoU {b.iou:.2f}, best candidate "
          f"{b.iou_oracle:.2f}, adapter {b.iou_ad:.2f}  [{len(s)} instances meet the rule]")
    print(f"  (c) ann {c.name}: {c.category}, {int(c.area_px)} px, frozen CP best candidate {c.iou_oracle:.2f}, "
          f"adapter {c.iou_ad:.2f}  [{len(l)} instances meet the rule; same tile as (b): {bool(len(same))}]")
    return b, c


def window(coco, img_dir, aid, r):
    a = coco.loadAnns([int(aid)])[0]
    info = coco.loadImgs([a["image_id"]])[0]
    try:
        import psutil; print(f"    free RAM {psutil.virtual_memory().available / 1e9:.1f} GB", flush=True)
    except Exception: pass
    print(f"    reading {info['file_name']} ({info['width']}x{info['height']}) ...", flush=True)
    img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, info["file_name"])), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    print("    ground truth and prompts ...", flush=True)
    wx, wy, ww, wh = int(r.win_x), int(r.win_y), int(r.win_w), int(r.win_h)
    g, gx0, gy0 = ev.gt_local(a["segmentation"], a["bbox"], H, W)
    gt = np.zeros((wh, ww), bool); gt[gy0 - wy:gy0 - wy + g.shape[0], gx0 - wx:gx0 - wx + g.shape[1]] = g
    P = ev.build_all_prompts(a, H, W, g, gx0, gy0)
    return dict(a=a, img=img, H=H, W=W, wx=wx, wy=wy, ww=ww, wh=wh, gt=gt,
                win=np.ascontiguousarray(img[wy:wy + wh, wx:wx + ww]),
                bb=ev.shift_prompt(P["BB"], wx, wy), cp=ev.shift_prompt(P["CP"], wx, wy))


def candidates(pred, o):
    masks, scores, _ = pred.predict(point_coords=o["cp"]["points"], point_labels=o["cp"]["labels"], box=None,
                                    multimask_output=True)
    masks = np.asarray(masks) > 0.5
    ious = [(m & o["gt"]).sum() / max(1, (m | o["gt"]).sum()) for m in masks]
    return masks, np.asarray(scores, float), np.array(ious)


def show_mask(ax, o, m, crop, txt, frame=None, title=None):
    x0, y0, side = crop
    sl = (slice(y0, y0 + side), slice(x0, x0 + side))
    ax.imshow(mq.overlay(o["win"][sl], o["gt"][sl], m[sl]), interpolation="nearest")
    mq.draw_prompt(ax, o["cp"], x0, y0)
    ax.set_xlim(-0.5, side - 0.5); ax.set_ylim(side - 0.5, -0.5)
    ax.text(0.5, 0.03, txt, transform=ax.transAxes, fontsize=6.3, color="white", ha="center", va="bottom",
            path_effects=OUT, linespacing=1.1)
    if title: ax.set_title(title, fontsize=7, pad=2.5)
    mq.clean(ax)
    if frame:
        for sp in ax.spines.values(): sp.set_color(frame); sp.set_linewidth(2.4)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); a_ = ap.parse_args()
    print("selecting examples ...")
    img_dir, _ = v3.resolve_img_dir(); coco = COCO(v3.ANN_FILE)
    b, c = pick(coco)
    import gc; gc.collect()
    ob, oc = window(coco, img_dir, b.name, b), window(coco, img_dir, c.name, c)
    print(f"  image {ob['W']}x{ob['H']}, window {ob['ww']}x{ob['wh']}", flush=True)
    res = {}
    for name, lora in (("frozen", None), ("adapted", LORA)):
        print(f"  loading {name} SAM ViT-H on {a_.device} ...", flush=True)
        pred = ev.load_model("sam1", a_.device, lora, "vit_h")
        for key, o in (("b", ob), ("c", oc)):
            print(f"    encoding window for ({key}) ...", flush=True)
            pred.set_image(o["win"]); res[(name, key)] = candidates(pred, o)
            m, s_, i = res[(name, key)]; ref = (b if key == "b" else c)
            ref_iou = ref.iou if name == "frozen" else ref.iou_ad
            print(f"  {name:<7} ({key}) predicted IoU {np.round(s_, 2)}  true IoU {np.round(i, 2)}  returned {i[int(np.argmax(s_))]:.2f}"
                  f"  (results.csv {ref_iou:.2f})")
        del pred
        import gc; gc.collect()

    FH = 6.0
    fig = plt.figure(figsize=(mf.FULL_W, FH))
    PA, PB = 0.27 * mf.FULL_W / FH, 0.20 * mf.FULL_W / FH          # square panel heights (figure fraction)
    def band(y0, y1, col, letter, title):
        fig.patches.append(Rectangle((0.005, y0), 0.99, y1 - y0, transform=fig.transFigure, fc=col, ec="none", zorder=-5))
        fig.text(0.015, y1 - 0.008, letter, fontsize=9, fontweight="bold", va="top")
        fig.text(0.04, y1 - 0.008, title, fontsize=8, va="top")

    # ---------------- (a)
    ya0 = 0.995 - 0.035 - 0.03 - PA
    band(ya0 - 0.012, 0.995, BAND["a"], "a", "Input protocol and prompts (iSAID validation image)")
    img, H, W = ob["img"], ob["H"], ob["W"]
    s = 1024.0 / max(H, W)
    small = cv2.resize(img, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
    ax = fig.add_axes([0.03, ya0, 0.30, PA]); ax.imshow(small); mq.clean(ax)
    tl = ya0 + PA + 0.006
    for x in ev.tile_starts(W, 1024, 768):
        ax.axvline(x * s, color="white", lw=0.4, alpha=0.8)
    for y in ev.tile_starts(H, 1024, 768):
        ax.axhline(y * s, color="white", lw=0.4, alpha=0.8)
    ax.add_patch(Rectangle((ob["wx"] * s, ob["wy"] * s), ob["ww"] * s, ob["wh"] * s, fill=False, ec=ORANGE, lw=1.4))
    side_b = np.sqrt(b.area_px)
    fig.text(0.18, tl, f"Whole image, resized to 1024 px\n(object {side_b * s:.0f} px at the input)", fontsize=6.6, ha="center", va="bottom")
    ax2 = fig.add_axes([0.37, ya0, 0.27, PA]); ax2.imshow(ob["win"]); mq.clean(ax2)
    for sp in ax2.spines.values(): sp.set_color(ORANGE); sp.set_linewidth(1.4)
    bx, by, bw, bh = ob["a"]["bbox"]; cr_b = mq.crop_box(bx - ob["wx"], by - ob["wy"], bw, bh, ob["ww"], ob["wh"], factor=1.9)
    ax2.add_patch(Rectangle((cr_b[0], cr_b[1]), cr_b[2], cr_b[2], fill=False, ec="white", lw=1.0,
                            path_effects=[pe.withStroke(linewidth=2.0, foreground="black")]))
    if c.image_id == b.image_id and c.win_x == b.win_x and c.win_y == b.win_y:
        cx, cy, cw, ch = oc["a"]["bbox"]
        ax2.add_patch(Rectangle((cx - oc["wx"] - 12, cy - oc["wy"] - 12), cw + 24, ch + 24, fill=False, ec="white", lw=0.9, ls="--"))
    fig.text(0.505, tl, f"1024-px tile, native resolution\n(object {side_b:.0f} px at the input)", fontsize=6.6, ha="center", va="bottom")
    ax3 = fig.add_axes([0.68, ya0, 0.27, PA])
    x0, y0, sd = cr_b; sl = (slice(y0, y0 + sd), slice(x0, x0 + sd))
    ax3.imshow(ob["win"][sl], interpolation="nearest")
    ax3.contour(ob["gt"][sl].astype(float), levels=[0.5], colors=[mq.C_TP / 255], linewidths=1.0)
    mq.draw_prompt(ax3, ob["bb"], x0, y0); mq.draw_prompt(ax3, ob["cp"], x0, y0)
    ax3.set_xlim(-0.5, sd - 0.5); ax3.set_ylim(sd - 0.5, -0.5); mq.clean(ax3)
    fig.text(0.815, tl, "Prompts from the ground truth\n(box and centroid)", fontsize=6.6, ha="center", va="bottom")
    for xa, xb in ((0.335, 0.365), (0.645, 0.675)):
        fig.patches.append(FancyArrowPatch((xa, ya0 + PA / 2), (xb, ya0 + PA / 2), transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=9, lw=0.9, color="0.25"))

    # ---------------- (b) and (c)
    BH = 0.035 + 0.028 + PB + 0.012
    y1 = ya0 - 0.012 - 0.008
    rows = [("b", ob, b, y1 - BH, y1, "Selection error: a correct candidate exists, but SAM returns another"),
            ("c", oc, c, y1 - 2 * BH - 0.008, y1 - BH - 0.008, "No correct candidate: every candidate leaks into the surroundings")]
    for key, o, r, y0b, y1b, title in rows:
        band(y0b, y1b, BAND[key], key, title + f" ({mq.nice_cat(r.category).lower()}, {int(r.area_px):,} px, centroid prompt)")
        bx, by, bw, bh = o["a"]["bbox"]
        crop = mq.crop_box(bx - o["wx"], by - o["wy"], bw, bh, o["ww"], o["wh"],
                           min_side=96 if key == "c" else 64, factor=6.0 if key == "c" else 2.2)
        m, sc, io = res[("frozen", key)]; ret = int(np.argmax(sc))
        h = PB
        for j in range(3):
            ax = fig.add_axes([0.03 + j * 0.215, y0b + 0.012, 0.20, h])
            frame = "black" if j == ret else (GREEN if io[j] >= 0.5 else None)
            lab = "returned" if j == ret else ("correct, not returned" if io[j] >= 0.5 else "")
            show_mask(ax, o, m[j], crop, f"predicted {sc[j]:.2f} | true {io[j]:.2f}" + (f"\n{lab}" if lab else ""),
                      frame=frame, title=f"Frozen SAM, candidate {j + 1}")
        m2, sc2, io2 = res[("adapted", key)]; r2 = int(np.argmax(sc2))
        ax = fig.add_axes([0.725, y0b + 0.012, 0.20, h])
        show_mask(ax, o, m2[r2], crop, f"predicted {sc2[r2]:.2f} | true {io2[r2]:.2f}\nreturned", frame="black",
                  title="LoRA + IoU head")
        fig.patches.append(FancyArrowPatch((0.672, y0b + 0.012 + h / 2), (0.718, y0b + 0.012 + h / 2), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=9, lw=0.9, color="0.25"))

    handles = [Patch(fc=mq.C_TP / 255, ec="0.3", lw=0.4, label="True positive"),
               Patch(fc=mq.C_FP / 255, ec="0.3", lw=0.4, label="False positive"),
               Patch(fc=mq.C_FN / 255, ec="0.3", lw=0.4, label="False negative"),
               Line2D([], [], marker="*", ls="none", mfc="white", mec="black", ms=7, label="Centroid prompt"),
               Patch(fc="none", ec="black", lw=1.8, label="Returned mask"),
               Patch(fc="none", ec=GREEN, lw=1.8, label="Correct candidate (IoU ≥ 0.5)")]
    fig.legend(handles=handles, ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.005), fontsize=6.3,
               handlelength=1.4, columnspacing=0.9)
    old = os.path.join(mf.OUT, "Figure_overview")
    for ext in (".eps", ".tif", ".png"):
        if os.path.exists(old + ext) and not os.path.exists(old + "_diagram" + ext):
            shutil.copy2(old + ext, old + "_diagram" + ext)
    mf.save(fig, "Figure_overview")
    print("done")


if __name__ == "__main__":
    main()
