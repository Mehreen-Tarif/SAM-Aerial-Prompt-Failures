r"""
sam_eval_v4.py -- ONE evaluation pipeline for every model and input mode.

Design:
  * SAM v1 and SAM 2 must use IDENTICAL prompts. v4 imports the prompt functions
    from sam_experiment_v3.py, so every strategy is built identically for both models.
  * Ground truth is rasterised on the image FILE's canvas (one iSAID image's annotation size differs from its file).
  * A ground-truth polygon that decodes to 0 pixels
    is excluded and logged (ann 18117).
  * All THREE candidate masks are scored (IoU, SAM score, area), so failures can be split
    into "no good mask existed" vs "SAM picked the wrong one"; predicted area separates
    empty masks from misplaced masks.
  * Input modes make SAM's input scale an experimental factor:
        whole    - whole image, resized by SAM to 1024 px
        tile1024 - 1024 px native-resolution tiles (no downscaling of objects)
        tile512  - 512 px tiles, upscaled 2x by SAM's own resize
    Each object goes to the tile that fully contains its box plus the 40-px CPN margin
    (so every prompt lies inside the tile); objects that fit no tile get their own crop.
    The per-instance input scale (1024 / longest window side) is recorded.
  * New strategy EXT: 4 extreme points (top/bottom/left/right-most mask pixels) + centroid,
    all inside the object (unlike CPC, whose corner points lie off the object).

Usage (from the repository folder):
    python sam_eval_v4.py --model sam1 --mode whole    --max-images 5      # smoke test
    python sam_eval_v4.py --model sam1 --mode tile1024
    python sam_eval_v4.py --model sam2 --mode whole
    python sam_eval_v4.py --model sam1 --mode tile1024 --strategies BB,CP  # subset
Resumable: re-run the same command after an interruption.
Output: results_v4\<model>_<mode>\results.csv  (+ log.txt, excluded.txt, done.txt)
"""
import os, json, sys, time, argparse, csv, gc, math
import numpy as np, cv2

B = os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(B, "scripts"))
import sam_experiment_v3 as v3
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

SAM2_REPO = os.environ.get("SAM2_REPO", os.path.join(os.path.dirname(B), "sam2"))
# optional overrides (defaults = the published runs). SAM 2.1 example:
#   set SAM2_CKPT=<sam2 repo>\checkpoints\sam2.1_hiera_large.pt
#   set SAM2_CFG=configs/sam2.1/sam2.1_hiera_l.yaml
#   set SAM2_NAME=sam2.1
SAM2_CKPT = os.environ.get("SAM2_CKPT", os.path.join(SAM2_REPO, "checkpoints", "sam2_hiera_large.pt"))
SAM2_CFG = os.environ.get("SAM2_CFG", "configs/sam2/sam2_hiera_l.yaml")
SAM2_NAME = os.environ.get("SAM2_NAME", "sam2")
# external dataset (COCO polygons), e.g. NWPU VHR-10:  set EVAL_ANN=...annotations.json  &  set EVAL_IMG_DIR=...images
EVAL_ANN = os.environ.get("EVAL_ANN", "")
EVAL_IMG_DIR = os.environ.get("EVAL_IMG_DIR", "")
BASE_STRATS = v3.STRATEGIES + ["EXT"]
NOISE_STRATS = ["BBN5", "BBN10", "BBN20", "CPJ5", "CPJ10", "CPJ20"]
ALL_STRATS = BASE_STRATS + NOISE_STRATS + ["DET"]
# DET: box from an object detector, matched to the ground-truth instance beforehand (det_boxes.py).
#   set DET_BOXES=...\det_boxes.json   (instances without a matched detection are skipped for DET)
DET_BOXES = os.environ.get("DET_BOXES", "")
CKPTS = {"vit_h": "sam_vit_h_4b8939.pth", "vit_l": "sam_vit_l_0b3195.pth", "vit_b": "sam_vit_b_01ec64.pth"}
NEG_MARGIN = 40
FIELDS = (["ann_id", "image_id", "category", "area_px", "stratum", "model", "mode", "strategy",
           "win_x", "win_y", "win_w", "win_h", "win_kind", "input_scale",
           "iou", "precision", "recall", "f1", "confidence", "pred_area", "chosen",
           "iou_m0", "iou_m1", "iou_m2", "score_m0", "score_m1", "score_m2",
           "area_m0", "area_m1", "area_m2", "iou_oracle", "time_ms", "error", "was_exception"])

# ------------------------------------------------------------------ prompts
def get_extreme_points(mask):
    ys, xs = np.where(mask)
    out = []
    for sel, axis in [(ys == ys.min(), xs), (ys == ys.max(), xs), (xs == xs.min(), ys), (xs == xs.max(), ys)]:
        v = np.sort(axis[sel]); m = v[len(v) // 2]
        out.append([m, ys[sel][0]] if axis is xs else [xs[sel][0], m])
    return np.array(out, dtype=float)

def build_all_prompts(ann, H, W, g, gx0, gy0):
    """Exactly v3's prompts for the 8 original strategies (global image coordinates; like v3,
    these use only the FIRST polygon of multi-part objects). EXT is new and is built from the
    FULL merged GT mask g (all parts): 4 extreme points + the full-mask centroid."""
    local_mask, x_off, y_off = v3.decode_local_mask(ann["segmentation"], ann["bbox"], H, W)
    prompts, _ = v3.build_prompts(local_mask, ann["bbox"], x_off, y_off)
    ys, xs = np.where(g)
    ext = np.vstack([get_extreme_points(g), [[xs.mean(), ys.mean()]]]) + np.array([gx0, gy0], dtype=float)
    prompts["EXT"] = {"points": ext, "labels": np.ones(5, dtype=int), "box": None}
    prompts["_first_polygon_empty"] = not local_mask.any()
    x, y, w, h = ann["bbox"]
    for px in (5, 10, 20):
        n = np.random.RandomState(ann["id"] + px).uniform(-px, px, 4)        # same draw as bpc_ablations.py
        xs_ = np.clip(sorted((x + n[0], x + w + n[2])), 0, W - 1); ys_ = np.clip(sorted((y + n[1], y + h + n[3])), 0, H - 1)
        if xs_[1] - xs_[0] < 1: xs_[1] = min(W - 1, xs_[0] + 1)
        if ys_[1] - ys_[0] < 1: ys_[1] = min(H - 1, ys_[0] + 1)
        prompts[f"BBN{px}"] = {"points": None, "labels": None, "box": np.array([xs_[0], ys_[0], xs_[1], ys_[1]], float)}
        m = np.random.RandomState(ann["id"] + 1000 + px).uniform(-px, px, 2)
        c = prompts["CP"]["points"][0] + m
        c = np.array([np.clip(c[0], 0, W - 1), np.clip(c[1], 0, H - 1)])
        prompts[f"CPJ{px}"] = {"points": c[None, :], "labels": np.ones(1, dtype=int), "box": None}
    return prompts

def scale_prompt(p, sx, sy):
    q = dict(p)
    if p["points"] is not None: q["points"] = p["points"] * np.array([sx, sy])
    if p["box"] is not None: q["box"] = p["box"] * np.array([sx, sy, sx, sy])
    return q

def legacy_gt(g, gx0, gy0, H, W, S=1024):
    """Exactly cv2.resize(full_mask, (S, S), interpolation=INTER_NEAREST) of the native GT mask,
    computed from the local mask only (dst pixel d samples src floor(d * src/S))."""
    src_y = np.minimum(H - 1, np.floor(np.arange(S) * (H / S)).astype(int))
    src_x = np.minimum(W - 1, np.floor(np.arange(S) * (W / S)).astype(int))
    R = np.where((src_y >= gy0) & (src_y < gy0 + g.shape[0]))[0]
    C = np.where((src_x >= gx0) & (src_x < gx0 + g.shape[1]))[0]
    out = np.zeros((S, S), bool)
    if len(R) and len(C): out[np.ix_(R, C)] = g[np.ix_(src_y[R] - gy0, src_x[C] - gx0)]
    return out

def shift_prompt(p, dx, dy):
    q = dict(p)
    if p["points"] is not None: q["points"] = p["points"] - np.array([dx, dy], dtype=float)
    if p["box"] is not None: q["box"] = p["box"] - np.array([dx, dy, dx, dy], dtype=float)
    return q

# ------------------------------------------------------------------ windows
def tile_starts(n, T, stride):
    return [0] if n <= T else sorted(set(list(range(0, n - T, stride)) + [n - T]))

def assign_windows(anns, H, W, mode):
    """Return {window_key: (x, y, w, h, kind)} and {ann_id: window_key}."""
    if mode == "whole":
        return {("whole",): (0, 0, W, H, "whole")}, {a["id"]: ("whole",) for a in anns}
    T = int(mode.replace("tile", "")); stride = T - T // 4
    xs, ys = tile_starts(W, T, stride), tile_starts(H, T, stride)
    wins, amap = {}, {}
    for a in anns:
        x, y, w, h = a["bbox"]
        need = (max(0, x - NEG_MARGIN), max(0, y - NEG_MARGIN), min(W, x + w + NEG_MARGIN), min(H, y + h + NEG_MARGIN))
        best = None
        for tx in xs:
            tw = min(T, W - tx)
            if not (need[0] >= tx and need[2] <= tx + tw): continue
            for ty in ys:
                th = min(T, H - ty)
                if need[1] >= ty and need[3] <= ty + th:
                    d = abs(x + w / 2 - tx - tw / 2) + abs(y + h / 2 - ty - th / 2)
                    if best is None or d < best[0]: best = (d, tx, ty, tw, th)
        if best:
            key = ("tile", best[1], best[2]); wins[key] = (best[1], best[2], best[3], best[4], "tile")
        else:   # own crop: square window around the object, at least T, clipped to the image
            S = max(T, int(math.ceil(max(need[2] - need[0], need[3] - need[1]) * 1.1)))
            cw, ch = min(S, W), min(S, H)
            cx0 = int(round(x + w / 2 - cw / 2)); cy0 = int(round(y + h / 2 - ch / 2))
            cx0 = min(max(0, cx0), W - cw); cy0 = min(max(0, cy0), H - ch)
            cx0 = min(cx0, int(math.floor(need[0]))); cy0 = min(cy0, int(math.floor(need[1])))
            cw = max(cw, int(math.ceil(need[2])) - cx0); ch = max(ch, int(math.ceil(need[3])) - cy0)
            key = ("crop", a["id"]); wins[key] = (cx0, cy0, min(cw, W - cx0), min(ch, H - cy0), "crop")
        amap[a["id"]] = key
    return wins, amap

def gt_local(seg, bbox, H, W, margin=3):
    """GT mask decoded only inside its box (+margin), ALL polygons merged, pycocotools' own rasteriser.
    Returns (mask, x0, y0) in image coordinates."""
    x, y, w, h = bbox
    x0, y0 = max(0, int(x - margin)), max(0, int(y - margin))
    x1, y1 = min(W, int(math.ceil(x + w)) + margin), min(H, int(math.ceil(y + h)) + margin)
    shifted = [(np.array(p, dtype=np.float64).reshape(-1, 2) - [x0, y0]).flatten().tolist() for p in seg]
    r = maskUtils.frPyObjects(shifted, y1 - y0, x1 - x0)
    r = maskUtils.merge(r) if isinstance(r, list) else r
    return maskUtils.decode(r).astype(bool), x0, y0

def fast_metrics(m, g, gx, gy, gt_area):
    """Exactly v3.compute_metrics_rle's quantities, without encoding the full-size mask:
    the intersection can only lie inside the GT box, the predicted area is a pixel count."""
    pa = float(np.count_nonzero(m))
    inter = float(np.count_nonzero(m[gy:gy + g.shape[0], gx:gx + g.shape[1]] & g))
    union = pa + gt_area - inter
    iou = inter / union if union > 0 else 0.0
    prec = inter / (pa + 1e-8); rec = inter / (gt_area + 1e-8)
    return iou, prec, rec, 2 * prec * rec / (prec + rec + 1e-8), pa

# ------------------------------------------------------------------ models
def load_model(name, device, lora=None, backbone="vit_h"):
    if name == "sam1":
        from segment_anything import sam_model_registry, SamPredictor
        ck = os.path.join(os.path.dirname(v3.SAM_CKPT), CKPTS[backbone])
        sam = sam_model_registry[backbone](checkpoint=ck).to(device)
        print(f"SAM {backbone} loaded from {ck}")
        if lora:
            sys.path.insert(0, B); from lora_sam import inject_lora, load_lora
            inject_lora(sam); sam.to(device); ck = load_lora(sam, lora); sam.eval()
            print(f"LoRA loaded from {lora} (epoch {ck.get('epoch', '?')})")
        return SamPredictor(sam)
    if lora or backbone != "vit_h": raise SystemExit("--lora and --backbone are only supported for --model sam1")
    class FastSAM2:
        """Same masks and scores as SAM2ImagePredictor.predict, but returns the thresholded masks as
        bool instead of converting them to full-resolution float32 (4x less GPU->CPU traffic, no re-threshold)."""
        def __init__(self, p): self.p = p
        def set_image(self, img): self.p.set_image(img)
        def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
            if not (hasattr(self.p, "_prep_prompts") and hasattr(self.p, "_predict")):
                return self.p.predict(point_coords=point_coords, point_labels=point_labels, box=box, multimask_output=multimask_output)
            mi, uc, lb, ub = self.p._prep_prompts(point_coords, point_labels, box, None, True)
            masks, ious, _ = self.p._predict(uc, lb, ub, mi, multimask_output, return_logits=False)
            return masks.squeeze(0).cpu().numpy(), ious.squeeze(0).float().cpu().numpy(), None
    sys.path.insert(0, SAM2_REPO); cwd = os.getcwd(); os.chdir(SAM2_REPO)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    pred = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_CKPT, device=device)); os.chdir(cwd)
    return pred if os.environ.get("SAM2_SLOW") else FastSAM2(pred)

def infer_ctx(name):
    if name == "sam2":
        import torch; return torch.inference_mode()
    import contextlib; return contextlib.nullcontext()

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sam1", "sam2"], required=True)
    ap.add_argument("--mode", default="whole", help="whole | tile1024 | tile512 | tile<N>")
    ap.add_argument("--strategies", default=",".join(BASE_STRATS),
                    help="default: the 9 main strategies; noisy prompts: BBN5,BBN10,BBN20,CPJ5,CPJ10,CPJ20")
    ap.add_argument("--backbone", default="vit_h", choices=list(CKPTS), help="SAM v1 backbone (sam1 only)")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(B, "results_v4"))
    ap.add_argument("--lora", default="", help="LoRA checkpoint (from lora_train_v3.py) to evaluate, sam1 only")
    ap.add_argument("--tag", default="", help="name for the LoRA run, used in the output folder")
    args = ap.parse_args()
    strats = [s.strip() for s in args.strategies.split(",") if s.strip()]
    assert all(s in ALL_STRATS for s in strats), f"unknown strategy in {strats}"
    model_name = (SAM2_NAME if args.model == "sam2" else args.model) + (f"-{args.backbone}" if args.backbone != "vit_h" else "") + (f"lora-{args.tag or 'x'}" if args.lora else "")
    legacy = args.mode == "legacy1024"   # the ORIGINAL submission's protocol (SAM_FINAL.py), on any population
    out_dir = os.path.join(args.out, f"{model_name}_{args.mode}"); os.makedirs(out_dir, exist_ok=True)
    res_p, done_p = os.path.join(out_dir, "results.csv"), os.path.join(out_dir, "done.txt")
    log = open(os.path.join(out_dir, "log.txt"), "a", encoding="utf-8")
    excl = open(os.path.join(out_dir, "excluded.txt"), "a", encoding="utf-8")
    def say(s): print(s); log.write(s + "\n"); log.flush()

    v3.set_determinism(v3.SEED)
    img_dir = EVAL_IMG_DIR or v3.resolve_img_dir()[0]
    coco = COCO(EVAL_ANN or v3.ANN_FILE)
    if EVAL_ANN: say(f"EXTERNAL DATASET: {EVAL_ANN}  images: {img_dir}")
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    DET = {}
    if "DET" in strats:
        if not DET_BOXES: raise SystemExit("strategy DET needs the environment variable DET_BOXES (det_boxes.json)")
        DET = {int(k): v["box"] for k, v in json.load(open(DET_BOXES)).items()}
        say(f"detector boxes: {len(DET):,} matched instances from {DET_BOXES}")
    done = set(int(l) for l in open(done_p)) if os.path.exists(done_p) else set()
    say(f"== {time.strftime('%Y-%m-%d %H:%M:%S')}  model={model_name} mode={args.mode} strategies={strats}  resume: {len(done)} done")
    if os.path.exists(res_p):   # drop rows of instances not marked done (killed mid-write) -> no duplicates on resume
        with open(res_p, newline="", encoding="utf-8") as f:
            rows_in = list(csv.DictReader(f))
        keep = [r for r in rows_in if int(r["ann_id"]) in done]
        if len(keep) != len(rows_in):
            with open(res_p, "w", newline="", encoding="utf-8") as f:
                w0 = csv.DictWriter(f, fieldnames=FIELDS); w0.writeheader(); w0.writerows(keep)
            say(f"resume cleanup: removed {len(rows_in) - len(keep)} rows of unfinished instances")
        del rows_in, keep
    predictor = load_model(args.model, v3.DEVICE, args.lora or None, args.backbone)
    new_file = not os.path.exists(res_p)
    fo = open(res_p, "a", newline="", encoding="utf-8"); wr = csv.DictWriter(fo, fieldnames=FIELDS)
    if new_file: wr.writeheader()
    fd = open(done_p, "a")

    img_ids = sorted(coco.getImgIds())
    if args.max_images: img_ids = img_ids[:args.max_images]
    t_start, n_inst, n_calls = time.time(), 0, 0
    for k, img_id in enumerate(img_ids, 1):
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])) if a["id"] not in done]
        if not anns: continue
        info = coco.loadImgs([img_id])[0]
        img_bgr = cv2.imread(os.path.join(img_dir, info["file_name"]))
        if img_bgr is None:
            say(f"!! could not read {info['file_name']}"); continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); H, W = img.shape[:2]
        if (W, H) != (info["width"], info["height"]):
            say(f"note: {info['file_name']} JSON size {info['width']}x{info['height']} != file {W}x{H}; using file size")
        valid, full_rle, full_area = [], {}, {}
        for a in anns:
            x, y, w, h = a["bbox"]
            full = maskUtils.frPyObjects(a["segmentation"], H, W)
            full = maskUtils.merge(full) if isinstance(full, list) else full
            if w <= 0 or h <= 0 or not a.get("segmentation") or not isinstance(a["segmentation"], list) or maskUtils.area(full) == 0:
                excl.write(f"{a['id']}\timage {img_id}\tbbox {a['bbox']}\tarea_field {a.get('area')}\tdecoded_area 0 or bad bbox\n")
                excl.flush(); fd.write(f"{a['id']}\n"); continue
            full_rle[a["id"]] = full; full_area[a["id"]] = float(maskUtils.area(full)); valid.append(a)
        wins, amap = assign_windows(valid, H, W, "whole" if legacy else args.mode)
        for key, (wx, wy, ww, wh, kind) in wins.items():
            members = [a for a in valid if amap[a["id"]] == key]
            if legacy:   # SAM_FINAL.py: cv2.resize(img, (1024, 1024)) -- square, aspect ratio NOT preserved
                sx, sy = 1024.0 / W, 1024.0 / H; kind = "legacy"
                win_img = cv2.resize(img, (1024, 1024)); scale = min(sx, sy)
            else:
                win_img = np.ascontiguousarray(img[wy:wy + wh, wx:wx + ww]); scale = 1024.0 / max(ww, wh)
            with infer_ctx(args.model):
                predictor.set_image(win_img)
            for a in members:
                area = int(a.get("area") or full_area[a["id"]])   # external datasets may lack "area"
                g, gx0, gy0 = gt_local(a["segmentation"], a["bbox"], H, W)
                gt_area = float(np.count_nonzero(g))
                if gt_area != full_area[a["id"]]:
                    say(f"  !! local GT area {gt_area} != full-canvas area {full_area[a['id']]} for ann {a['id']} -- using full canvas")
                    g = maskUtils.decode(full_rle[a["id"]]).astype(bool); gx0 = gy0 = 0; gt_area = float(full_area[a["id"]])
                gx, gy = gx0 - wx, gy0 - wy
                if not legacy and (gx < 0 or gy < 0 or gx + g.shape[1] > ww or gy + g.shape[0] > wh):
                    raise RuntimeError(f"GT box of ann {a['id']} not inside its window")
                g_eval, gx_e, gy_e, ga_e = g, gx, gy, gt_area
                if legacy:   # GT downsampled to 1024x1024 with INTER_NEAREST, IoU on that canvas (old protocol)
                    g_eval = legacy_gt(g, gx0, gy0, H, W); gx_e = gy_e = 0; ga_e = float(np.count_nonzero(g_eval))
                try:
                    prompts = build_all_prompts(a, H, W, g, gx0, gy0)
                    if prompts["_first_polygon_empty"]:
                        say(f"  note: ann {a['id']} first polygon decodes to 0 px (v3 prompts use it -> NaN points, kept as in v3)")
                except Exception as e:
                    say(f"  !! prompt construction failed for ann {a['id']}: {str(e)[:120]} -> scored 0, flagged")
                    prompts = None
                base = dict(ann_id=a["id"], image_id=img_id, category=cats[a["category_id"]], area_px=area,
                            stratum=v3.stratum_of(area), model=model_name, mode=args.mode, win_x=wx, win_y=wy,
                            win_w=ww, win_h=wh, win_kind=kind, input_scale=round(scale, 5))
                if prompts is not None and a["id"] in DET:
                    prompts["DET"] = {"points": None, "labels": None, "box": np.array(DET[a["id"]], dtype=float)}
                for s in strats:
                    if s == "DET" and a["id"] not in DET: continue      # not detected: counted as a miss in the report
                    row = dict(base, strategy=s)
                    t0 = time.perf_counter()
                    try:
                        if prompts is None: raise RuntimeError("prompt construction failed")
                        p = scale_prompt(prompts[s], sx, sy) if legacy else shift_prompt(prompts[s], wx, wy)
                        with infer_ctx(args.model):
                            masks, scores, _ = predictor.predict(point_coords=p["points"], point_labels=p["labels"],
                                                                 box=p["box"], multimask_output=True)
                        ms = (time.perf_counter() - t0) * 1000
                        masks = np.asarray(masks) > 0.5 if np.asarray(masks).dtype != bool else np.asarray(masks)
                        scores = np.asarray(scores, dtype=float); best = int(np.argmax(scores))
                        mets = [fast_metrics(m, g_eval, gx_e, gy_e, ga_e) for m in masks]
                        ious = np.array([q[0] for q in mets]); areas = np.array([q[4] for q in mets])
                        iou, prec, rec, f1 = mets[best][:4]
                        row.update(iou=iou, precision=prec, recall=rec, f1=f1, confidence=float(scores[best]),
                                   pred_area=float(areas[best]), chosen=best, iou_oracle=float(ious.max()),
                                   time_ms=round(ms, 3), error="", was_exception=False)
                        for j in range(3):
                            row[f"iou_m{j}"] = float(ious[j]) if j < len(ious) else np.nan
                            row[f"score_m{j}"] = float(scores[j]) if j < len(scores) else np.nan
                            row[f"area_m{j}"] = float(areas[j]) if j < len(areas) else np.nan
                        if iou < 0: raise RuntimeError("negative IoU (canvas mismatch)")
                    except Exception as e:   # a failure is a data point: scored 0 and flagged
                        row.update(iou=0.0, precision=0.0, recall=0.0, f1=0.0, confidence=np.nan, pred_area=np.nan,
                                   chosen=-1, iou_oracle=0.0, time_ms=np.nan, error=str(e)[:120], was_exception=True)
                        say(f"  exception ann {a['id']} {s}: {str(e)[:120]}")
                    wr.writerow(row); n_calls += 1
                fo.flush(); fd.write(f"{a['id']}\n"); fd.flush(); n_inst += 1
        el = time.time() - t_start
        say(f"[{k}/{len(img_ids)}] image {img_id} {W}x{H}: {len(valid)} inst, {len(wins)} windows | "
            f"total {n_inst:,} inst, {el/60:.1f} min")
        del img, img_bgr; gc.collect()
        if k % 5 == 0:
            try:
                import torch; torch.cuda.empty_cache()
            except Exception: pass
    say(f"== finished: {n_inst:,} instances, {n_calls:,} predictions in {(time.time()-t_start)/60:.1f} min")

if __name__ == "__main__":
    main()
