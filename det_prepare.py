r"""
det_prepare.py -- builds a YOLO-format detection dataset from the iSAID TRAINING split (CPU, ~20-40 min).

The detector is trained only on iSAID train images, with the same held-out images as the adapters
(seed 42, 5%), so the iSAID validation images used for testing are never seen.
  tiles: 1024 x 1024, stride 768 (as in the evaluation); horizontal boxes from the instance annotations
  a box is kept in a tile if at least half of its area lies inside (then clipped to the tile)
  tiles without objects: 10% kept (seeded) as background examples

Run from the repository folder:  python det_prepare.py            (output: D:\iSAID_Data\yolo_isaid, ~6-10 GB)
"""
import os, sys, random, json, shutil
import numpy as np, cv2

B = os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, B); sys.path.insert(0, os.path.join(B, "scripts"))
import sam_experiment_v3 as v3
from pycocotools.coco import COCO

OUT = os.environ.get("YOLO_DIR", os.path.join(v3.DATA_DIR, "yolo_isaid"))
TRAIN_ANN = os.path.join(v3.DATA_DIR, "data", "iSAID", "Annotations", "iSAID_train_20190823_114751.json")
TRAIN_IMG = os.path.join(v3.DATA_DIR, "data", "iSAID", "train", "images")
T, STRIDE, KEEP_EMPTY, SEED, VAL_FRAC = 1024, 768, 0.10, 42, 0.05


def starts(n):
    return [0] if n <= T else sorted(set(list(range(0, n - T, STRIDE)) + [n - T]))


def main():
    coco = COCO(TRAIN_ANN)
    cat_ids = sorted(coco.getCatIds()); cidx = {c: i for i, c in enumerate(cat_ids)}
    names = [coco.loadCats([c])[0]["name"] for c in cat_ids]
    ids = sorted(coco.getImgIds()); rng = random.Random(SEED); rng.shuffle(ids)     # identical split to the adapters
    n_val = max(3, int(round(VAL_FRAC * len(ids))))
    split = {i: "val" for i in ids[:n_val]}; split.update({i: "train" for i in ids[n_val:]})
    if os.path.exists(OUT): shutil.rmtree(OUT)
    for s in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", s)); os.makedirs(os.path.join(OUT, "labels", s))
    erng = random.Random(SEED + 7); stats = {"train": [0, 0], "val": [0, 0]}; nbytes = 0
    for k, img_id in enumerate(sorted(ids), 1):
        info = coco.loadImgs([img_id])[0]; s = split[img_id]
        img = cv2.imread(os.path.join(TRAIN_IMG, info["file_name"]))
        if img is None: print("!! cannot read", info["file_name"]); continue
        H, W = img.shape[:2]
        boxes = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            x, y, w, h = a["bbox"]
            if w > 0 and h > 0 and a.get("segmentation"): boxes.append((cidx[a["category_id"]], x, y, x + w, y + h))
        boxes = np.array(boxes, float).reshape(-1, 5)
        for ty in starts(H):
            for tx in starts(W):
                tw, th = min(T, W - tx), min(T, H - ty)
                if len(boxes):
                    x0 = np.clip(boxes[:, 1], tx, tx + tw); x1 = np.clip(boxes[:, 3], tx, tx + tw)
                    y0 = np.clip(boxes[:, 2], ty, ty + th); y1 = np.clip(boxes[:, 4], ty, ty + th)
                    inside = (x1 - x0) * (y1 - y0) / ((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2]))
                    keep = (inside >= 0.5) & (x1 - x0 >= 1) & (y1 - y0 >= 1)
                else:
                    keep = np.zeros(0, bool)
                if not keep.any() and erng.random() > KEEP_EMPTY: continue
                stem = f"{os.path.splitext(info['file_name'])[0]}_{tx}_{ty}"
                ip = os.path.join(OUT, "images", s, stem + ".jpg")
                cv2.imwrite(ip, img[ty:ty + th, tx:tx + tw], [cv2.IMWRITE_JPEG_QUALITY, 95]); nbytes += os.path.getsize(ip)
                with open(os.path.join(OUT, "labels", s, stem + ".txt"), "w") as f:
                    for j in np.where(keep)[0]:
                        cx = ((x0[j] + x1[j]) / 2 - tx) / tw; cy = ((y0[j] + y1[j]) / 2 - ty) / th
                        f.write(f"{int(boxes[j, 0])} {cx:.6f} {cy:.6f} {(x1[j] - x0[j]) / tw:.6f} {(y1[j] - y0[j]) / th:.6f}\n")
                stats[s][0] += 1; stats[s][1] += int(keep.sum())
        if k % 50 == 0: print(f"[{k}/{len(ids)}] tiles {stats}  {nbytes / 1e9:.1f} GB")
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, n in enumerate(names): f.write(f"  {i}: {n}\n")
    json.dump(dict(names=names, cat_ids=cat_ids, split_seed=SEED, val_images=n_val, tiles=stats),
              open(os.path.join(OUT, "dataset_info.json"), "w"), indent=2)
    print(f"done: {stats} (tiles, boxes)  {nbytes / 1e9:.1f} GB -> {OUT}")


if __name__ == "__main__":
    main()
