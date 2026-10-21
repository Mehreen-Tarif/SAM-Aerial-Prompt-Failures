r"""
det_boxes.py -- runs the trained detector on the iSAID VALIDATION images and matches its boxes to the instances.

  detection: 1024-px tiles, stride 768 (as in the evaluation), confidence >= 0.05, class-wise NMS (IoU 0.6) across tiles
  operating point, fixed in advance: confidence >= 0.25 (Ultralytics default)
  matching (COCO style): detections in descending confidence; each takes the unmatched ground-truth instance of the
      SAME class with the highest box IoU, if that IoU >= 0.5
Output (in det\):  det_boxes.json  {ann_id: {box [x0,y0,x1,y1], conf, box_iou}}   -> used by sam_eval_v4.py (strategy DET)
                   det_all.json    every detection with confidence >= 0.05 (for later analysis)
                   det_report.txt  recall / precision per size stratum and the detector's box errors in pixels
Run from the repository folder:  python det_boxes.py           (GPU, ~20-40 min)
"""
import os, sys, json, time
import numpy as np, cv2, torch, torchvision

B = os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, B); sys.path.insert(0, os.path.join(B, "scripts"))
import sam_experiment_v3 as v3
from pycocotools.coco import COCO

T, STRIDE, CONF_MIN, CONF_OP, NMS_IOU, MATCH_IOU = 1024, 768, 0.05, 0.25, 0.6, 0.5
OUT = os.path.join(B, "det")
WEIGHTS = os.environ.get("DET_WEIGHTS", os.path.join(OUT, "yolo11m_isaid", "weights", "best.pt"))


def starts(n):
    return [0] if n <= T else sorted(set(list(range(0, n - T, STRIDE)) + [n - T]))


def box_iou(a, b):
    """a: (n,4), b: (m,4) xyxy -> (n,m)"""
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def main():
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    rep = open(os.path.join(OUT, "det_report.txt"), "w", encoding="utf-8")
    def say(s=""): print(s); rep.write(str(s) + "\n"); rep.flush()
    model = YOLO(WEIGHTS)
    info_p = os.path.join(os.environ.get("YOLO_DIR", os.path.join(v3.DATA_DIR, "yolo_isaid")), "dataset_info.json")
    info = json.load(open(info_p))
    coco = COCO(v3.ANN_FILE); img_dir, _ = v3.resolve_img_dir()
    # map detector classes to VALIDATION category ids BY NAME (the train and val files number the categories differently)
    val_ids = {c["name"].lower(): c["id"] for c in coco.loadCats(coco.getCatIds())}
    cat_ids = [val_ids[n.lower()] for n in info["names"]]
    say("detector class -> validation category: " + ", ".join(f"{n} {t}->{v}" for n, t, v in zip(info["names"], info["cat_ids"], cat_ids)))
    say(f"detector {WEIGHTS}\noperating point conf >= {CONF_OP}, match IoU >= {MATCH_IOU} (same class)")
    matched, all_dets, rows, n_fp, agn_hit, agn_n = {}, {}, [], 0, 0, 0
    t0 = time.time(); ids = sorted(coco.getImgIds())
    for k, img_id in enumerate(ids, 1):
        info = coco.loadImgs([img_id])[0]
        img = cv2.imread(os.path.join(img_dir, info["file_name"]))
        if img is None: say(f"!! cannot read {info['file_name']}"); continue
        H, W = img.shape[:2]
        tiles = [(tx, ty) for ty in starts(H) for tx in starts(W)]
        bx, sc, cl = [], [], []
        for i in range(0, len(tiles), 8):
            batch = tiles[i:i + 8]
            res = model.predict([np.ascontiguousarray(img[ty:ty + T, tx:tx + T]) for tx, ty in batch], imgsz=T,
                                conf=CONF_MIN, iou=NMS_IOU, max_det=1500, verbose=False)
            for (tx, ty), r in zip(batch, res):
                if r.boxes is None or len(r.boxes) == 0: continue
                bx.append(r.boxes.xyxy.cpu().numpy() + [tx, ty, tx, ty]); sc.append(r.boxes.conf.cpu().numpy())
                cl.append(r.boxes.cls.cpu().numpy().astype(int))
        if bx:
            bx, sc, cl = (np.concatenate(bx).astype(np.float32), np.concatenate(sc).astype(np.float32),
                          np.concatenate(cl))
            keep = torchvision.ops.batched_nms(torch.as_tensor(bx), torch.as_tensor(sc), torch.as_tensor(cl), NMS_IOU).numpy()
            bx, sc, cl = bx[keep], sc[keep], cl[keep]
        else:
            bx, sc, cl = np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
        all_dets[img_id] = [[*map(float, b), float(s), int(cat_ids[c])] for b, s, c in zip(bx, sc, cl)]
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
                if a["bbox"][2] > 0 and a["bbox"][3] > 0 and isinstance(a.get("segmentation"), list)]
        op = sc >= CONF_OP
        gall = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]).reshape(-1, 4)
        if len(gall):
            agn_n += len(gall); agn_hit += int((box_iou(bx[op], gall).max(0) >= MATCH_IOU).sum()) if op.any() else 0
        for c in set(a["category_id"] for a in anns) | set(int(cat_ids[c]) for c in cl[op]):
            g = [a for a in anns if a["category_id"] == c]
            gb = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in g]).reshape(-1, 4)
            di = np.where(op & (np.array([cat_ids[x] for x in cl]) == c))[0] if len(cl) else np.zeros(0, int)
            di = di[np.argsort(-sc[di], kind="stable")]
            M = box_iou(bx[di], gb); used = np.zeros(len(g), bool)
            for r_, d in enumerate(di):
                if not len(g): break
                cand = np.where(~used)[0]
                if not len(cand): break
                j = cand[np.argmax(M[r_, cand])]
                if M[r_, j] >= MATCH_IOU:
                    used[j] = True
                    matched[g[j]["id"]] = dict(box=[float(v) for v in bx[d]], conf=float(sc[d]), box_iou=float(M[r_, j]))
            for j, a in enumerate(g):
                e = (np.array(matched[a["id"]]["box"]) - gb[j]) if used[j] else np.full(4, np.nan)
                rows.append((a["id"], v3.stratum_of(int(a.get("area", 0))), used[j], *e))
            n_fp += len(di) - int(used.sum())
        if k % 25 == 0: print(f"[{k}/{len(ids)}] matched {len(matched):,}  {(time.time() - t0) / 60:.1f} min")
    json.dump({str(k): v for k, v in matched.items()}, open(os.path.join(OUT, "det_boxes.json"), "w"))
    json.dump({str(k): v for k, v in all_dets.items()}, open(os.path.join(OUT, "det_all.json"), "w"))
    import pandas as pd
    D = pd.DataFrame(rows, columns=["ann_id", "stratum", "matched", "dx0", "dy0", "dx1", "dy1"])
    G = D; nfp = n_fp
    say(f"\n(sanity check, class-agnostic best box IoU >= 0.5 for {100 * agn_hit / max(1, agn_n):.1f}% of instances)")
    say(f"ground-truth instances {len(G):,}; matched {int(G.matched.sum()):,} (recall {100 * G.matched.mean():.1f}%); "
        f"unmatched detections at the operating point {nfp:,} (precision {100 * G.matched.sum() / max(1, G.matched.sum() + nfp):.1f}%)")
    for z in ["small", "medium", "large"]:
        g = G[G.stratum == z]; say(f"  {z:<6} n {len(g):>7,}  recall {100 * g.matched.mean():5.1f}%")
    E = G[G.matched][["dx0", "dy0", "dx1", "dy1"]].abs()
    say("\nbox error of matched detections, |offset| per edge in pixels (median / 90th percentile):")
    say(f"  all     {np.nanmedian(E.values):.1f} / {np.nanpercentile(E.values, 90):.1f}")
    for z in ["small", "medium", "large"]:
        e = G[G.matched & (G.stratum == z)][["dx0", "dy0", "dx1", "dy1"]].abs().values
        if len(e): say(f"  {z:<6}  {np.nanmedian(e):.1f} / {np.nanpercentile(e, 90):.1f}")
    if agn_hit and len(matched) < 0.5 * agn_hit:
        say("!! far fewer class-matched than class-agnostic hits: class mapping is wrong - stopping"); os.remove(os.path.join(OUT, "det_boxes.json")); sys.exit(2)
    say(f"\nwrote det_boxes.json ({len(matched):,} instances) and det_all.json -> {OUT}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
