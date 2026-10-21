"""
make_examples_v4.py  -  Figure_examples with a fourth column: centroid prompt after mixed-prompt adaptation.
The three instances are chosen by the same rule as before, using FROZEN SAM only (make_qualitative.pick_examples),
so the adapted model's result is shown whatever it is.

Run from the repository folder (after the GPU job has finished):
    python make_examples_v4.py --device cuda
Output: results_v4/figures_paper/Figure_examples.*  (EPS + 600-dpi TIFF + PNG) and the IoUs printed to screen
"""
import os, sys, argparse
import numpy as np
import cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_qualitative as mq
from make_qualitative import mf, ev, v3, plt, Patch, Line2D, pe, COCO


def predict(predictor, prompts, st, wx, wy, gt):
    p = ev.shift_prompt(prompts[st], wx, wy)
    masks, scores, _ = predictor.predict(point_coords=p["points"], point_labels=p["labels"], box=p["box"],
                                         multimask_output=True)
    m = np.asarray(masks)[int(np.argmax(scores))] > 0.5
    return p, m, (m & gt).sum() / max(1, (m | gt).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lora", default=os.path.join("lora_v4", "tile1024", "best.pt"))
    a_ = ap.parse_args()
    picks = mq.pick_examples()
    img_dir, _ = v3.resolve_img_dir()
    coco = COCO(v3.ANN_FILE)
    frozen = ev.load_model("sam1", a_.device, None, "vit_h")
    adapted = ev.load_model("sam1", a_.device, a_.lora, "vit_h")
    cells = []
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
        out = {"stratum": s, "cat": r.category, "area": int(r.area_px), "prompts": {}, "pred": {}, "iou": {}}
        frozen.set_image(win)
        for st in ["BB", "CP"]:
            out["prompts"][st], out["pred"][st], out["iou"][st] = predict(frozen, prompts, st, wx, wy, gt)
        adapted.set_image(win)
        out["prompts"]["CPa"], out["pred"]["CPa"], out["iou"]["CPa"] = predict(adapted, prompts, "CP", wx, wy, gt)
        print(f"  {s:<6} ann {aid}: BB {out['iou']['BB']:.2f}  CP {out['iou']['CP']:.2f}  CP adapted {out['iou']['CPa']:.2f}")
        bx, by, bw, bh = a["bbox"]
        out["crop"] = mq.crop_box(bx - wx, by - wy, bw, bh, ww, wh)
        out["win"], out["gt"] = win, gt
        cells.append(out)

    cols = [("GT", "Ground truth"), ("BB", "Box, frozen SAM"), ("CP", "Centroid, frozen SAM"),
            ("CPa", "Centroid, LoRA + head")]
    fig, axes = plt.subplots(3, 4, figsize=(mf.FULL_W, mf.FULL_W * 0.75 + 0.35))
    letters = iter("abcdefghijkl")
    for i, c in enumerate(cells):
        x0, y0, side = c["crop"]
        sl = (slice(y0, y0 + side), slice(x0, x0 + side))
        im, gt = c["win"][sl], c["gt"][sl]
        for j, (col, _) in enumerate(cols):
            ax = axes[i, j]
            if col == "GT":
                ax.imshow(mq.overlay(im, gt), interpolation="nearest"); mq.scale_bar(ax, side)
            else:
                ax.imshow(mq.overlay(im, gt, c["pred"][col][sl]), interpolation="nearest")
                mq.draw_prompt(ax, c["prompts"][col], x0, y0)
                ax.text(0.96, 0.04, f"IoU {c['iou'][col]:.2f}", transform=ax.transAxes, fontsize=7,
                        color="white", ha="right", va="bottom", path_effects=mq.OUTLINE)
            ax.set_xlim(-0.5, side - 0.5); ax.set_ylim(side - 0.5, -0.5)
            mq.tag(ax, next(letters)); mq.clean(ax)
        axes[i, 0].set_ylabel(f"{c['stratum'].capitalize()}\n{mq.nice_cat(c['cat'])}, {c['area']:,} px", fontsize=8)
    for j, (_, t) in enumerate(cols):
        axes[0, j].set_title(t, fontsize=8, pad=4)
    handles = [Patch(fc=mq.C_TP / 255, ec="0.3", lw=0.4, label="True positive / ground truth"),
               Patch(fc=mq.C_FP / 255, ec="0.3", lw=0.4, label="False positive"),
               Patch(fc=mq.C_FN / 255, ec="0.3", lw=0.4, label="False negative"),
               Line2D([], [], color="white", lw=1.1, label="Box prompt", path_effects=[pe.withStroke(linewidth=2.4, foreground="black")]),
               Line2D([], [], marker="*", ls="none", mfc="white", mec="black", ms=8, label="Point prompt")]
    fig.legend(handles=handles, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.02), fontsize=7)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    mf.save(fig, "Figure_examples")
    print("done")


if __name__ == "__main__":
    main()
