r"""
SAM PROMPT STRATEGY EXPERIMENT — EVALUATION PIPELINE (v3)
==========================================================
Changes from v2:
  - FIXED: images are no longer force-resized to a square 1024x1024
    before being passed to SAM. That approach stretched non-square
    images (DOTA/iSAID images range from 800x800 to 20,000x20,000
    with highly variable aspect ratios), distorting object shapes.
    SamPredictor.set_image() is designed to take the ORIGINAL image
    and handle aspect-ratio-preserving resize + padding internally.
    Prompt coordinates (points, boxes) are now passed directly in
    ORIGINAL image pixel space — SAM's predict() transforms them
    internally. This also removes the sx/sy manual scaling logic
    entirely, and removes the need to resize ground-truth masks,
    since predict() returns masks already in original resolution.
  - ADDED: cudnn determinism settings, so the seed=42 reproducibility
    claim actually holds on GPU, not just in the sampling logic.
  - ADDED: environment manifest (versions, GPU, timestamp) written
    alongside results, for the GitHub reproducibility release.
  - ADDED: a one-time sanity check on the CPN (negative point)
    strategy at the start of a run, confirming negative points
    actually land in the background as intended.

Everything else (per-image encoder caching, full val set / no
artificial cap, size-stratum tagging, checkpointing/resume, 8
strategies including CPN) is unchanged from v2.

HOW TO RUN
----------
    python sam_experiment_v3.py --smoke      # ~30 image sanity check
    python sam_experiment_v3.py               # full run, resumable
"""

import os
import sys
import time
import json
import platform
import argparse
import warnings
import gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
from segment_anything import sam_model_registry, SamPredictor
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

DATA_DIR = os.environ.get("ISAID_ROOT", r"D:\iSAID_Data")   # folder holding data\iSAID and the SAM checkpoint
ANN_FILE = os.path.join(DATA_DIR, "data", "iSAID", "Annotations",
                         "iSAID_val_20190823_114742.json")
IMG_DIR_CANDIDATES = [
    os.path.join(DATA_DIR, "data", "iSAID", "val", "images"),
    os.path.join(DATA_DIR, "data", "iSAID", "val"),
]
SAM_CKPT = os.path.join(DATA_DIR, "sam_vit_h_4b8939.pth")

PROJECT_DIR = os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_DIR, "results")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLEANUP_INTERVAL = 5  # clear CUDA memory cache every N images to prevent
                       # fragmentation-driven slowdown over long runs

SMALL_MAX = 32 * 32
MEDIUM_MAX = 96 * 96

RESUME_FILE = os.path.join(OUT_DIR, "checkpoint_done.txt")
RESULTS_FILE = os.path.join(OUT_DIR, "raw_results_full.csv")
LOG_FILE = os.path.join(OUT_DIR, "load_log.txt")
MANIFEST_FILE = os.path.join(OUT_DIR, "environment_manifest.json")

STRATEGIES = ["CP", "BB", "MP3", "MP5", "MP10", "BPC", "CPC", "CPN"]

STRATEGY_NAMES = {
    "CP": "Center Point", "BB": "Bounding Box", "MP3": "Multi-Point 3",
    "MP5": "Multi-Point 5", "MP10": "Multi-Point 10", "BPC": "Box+Center",
    "CPC": "Corners+Center", "CPN": "Center+3 Negative",
}

# ═══════════════════════════════════════════════════════════════════
# DETERMINISM — make the seed=42 reproducibility claim actually true
# ═══════════════════════════════════════════════════════════════════

def set_determinism(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_environment_manifest():
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    try:
        import segment_anything
        manifest["segment_anything_version"] = getattr(segment_anything, "__version__", "unknown (no __version__ attr)")
    except Exception:
        manifest["segment_anything_version"] = "unknown"
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Environment manifest written to:", MANIFEST_FILE)
    for k, v in manifest.items():
        print(f"    {k}: {v}")
    print()


# ═══════════════════════════════════════════════════════════════════
# SETUP / VALIDATION
# ═══════════════════════════════════════════════════════════════════

def resolve_img_dir():
    for cand in IMG_DIR_CANDIDATES:
        if os.path.isdir(cand):
            n = len([f for f in os.listdir(cand)
                      if f.lower().endswith((".png", ".jpg", ".tif"))])
            if n > 0:
                return cand, n
    return None, 0


def preflight():
    print("=" * 65)
    print("  PRE-FLIGHT CHECK")
    print("=" * 65)
    ok = True

    if not os.path.exists(ANN_FILE):
        print(f"  MISSING  annotation file: {ANN_FILE}")
        ok = False
    else:
        print(f"  OK       annotation file")

    img_dir, n_imgs = resolve_img_dir()
    if img_dir is None:
        print(f"  MISSING  could not find a val image folder with PNGs")
        ok = False
    else:
        print(f"  OK       image folder: {img_dir}  ({n_imgs} images)")
        if n_imgs < 400:
            print(f"  WARNING  only {n_imgs} images found — expected ~458 for the full iSAID val set.")
            print(f"           Results from a partial image set should NOT be treated as the")
            print(f"           full-val-set experiment. Confirm your image download is complete")
            print(f"           before running the full (non --smoke) experiment.")

    if not os.path.exists(SAM_CKPT):
        print(f"  MISSING  SAM checkpoint: {SAM_CKPT}")
        ok = False
    else:
        size_mb = os.path.getsize(SAM_CKPT) / (1024 * 1024)
        print(f"  OK       SAM checkpoint ({size_mb:.0f} MB)")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  OK       device: {DEVICE.upper()} ({gpu})")

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 65)
    if not ok:
        print("  Fix the issues above before continuing.")
        sys.exit(1)
    print("  Checks passed.\n")
    return img_dir


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def stratum_of(area_px):
    if area_px < SMALL_MAX:
        return "small"
    elif area_px < MEDIUM_MAX:
        return "medium"
    return "large"


def load_all_instances(img_dir):
    print("Loading iSAID annotations...")
    coco = COCO(ANN_FILE)

    cat_info = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in cat_info}
    print(f"  Found {len(cat_info)} categories:")
    for c in cat_info:
        print(f"    - {c['name']}")

    available_files = set(os.listdir(img_dir))
    all_ann_ids = coco.getAnnIds()
    print(f"  Total annotations: {len(all_ann_ids)}")

    # NOTE: uses the precomputed COCO 'area' field rather than decoding
    # every mask here. DOTA/iSAID images can be up to 20,000x20,000 px;
    # decoding a full-image-sized array for each of ~116,000 annotations
    # just to check validity/area took 48 minutes in testing. The actual
    # mask IS decoded later, lazily, per-instance inside run() — that's
    # unavoidable and correct there since the real mask is needed for
    # prompts/metrics. This loading step only needs a cheap validity check.
    instances = []
    skip_reasons = {"no_image": 0, "no_seg": 0, "zero_area": 0,
                     "bad_bbox": 0, "decode_error": 0}

    for aid in tqdm(all_ann_ids, desc="  Validating annotations", ncols=70):
        ann = coco.loadAnns([aid])[0]
        img_info = coco.loadImgs([ann["image_id"]])[0]
        fname = img_info["file_name"]
        if fname not in available_files:
            skip_reasons["no_image"] += 1
            continue

        seg = ann.get("segmentation")
        if not seg:
            skip_reasons["no_seg"] += 1
            continue

        area = ann.get("area", 0)
        if area <= 0:
            skip_reasons["zero_area"] += 1
            continue

        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            skip_reasons["bad_bbox"] += 1
            continue

        cat_name = cat_id_to_name.get(ann["category_id"], "unknown")

        instances.append({
            "ann_id": aid, "image_id": ann["image_id"], "file_name": fname,
            "category": cat_name, "bbox": ann["bbox"], "area_px": int(area),
            "stratum": stratum_of(int(area)),
            "img_h": img_info["height"], "img_w": img_info["width"],
        })

    with open(LOG_FILE, "w") as f:
        f.write(f"Total annotations: {len(all_ann_ids)}\n")
        f.write(f"Valid instances loaded: {len(instances)}\n")
        for k, v in skip_reasons.items():
            f.write(f"Skipped ({k}): {v}\n")
        from collections import Counter
        cat_counts = Counter(it["category"] for it in instances)
        strat_counts = Counter(it["stratum"] for it in instances)
        f.write("\nPer-category counts:\n")
        for k, v in sorted(cat_counts.items()):
            f.write(f"  {k}: {v}\n")
        f.write("\nPer-stratum counts:\n")
        for k, v in sorted(strat_counts.items()):
            f.write(f"  {k}: {v}\n")

    print(f"\n  Loaded {len(instances)} valid instances. Skipped: {skip_reasons}")
    print(f"  Full breakdown: {LOG_FILE}\n")
    return instances, coco


# ═══════════════════════════════════════════════════════════════════
# PROMPT GENERATORS — all in ORIGINAL image coordinate space now
# ═══════════════════════════════════════════════════════════════════

def get_center_point(mask):
    ys, xs = np.where(mask)
    return np.array([[float(xs.mean()), float(ys.mean())]])


def get_interior_points(mask, n):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(mask.astype(np.uint8), k, iterations=2)
    ys, xs = np.where(eroded > 0)
    if len(xs) < n:
        ys, xs = np.where(mask)
    if len(xs) == 0:
        return get_center_point(mask)
    idx = np.linspace(0, len(xs) - 1, n, dtype=int)
    return np.column_stack([xs[idx], ys[idx]]).astype(float)


def get_negative_points(mask, bbox, n=3, margin=40):
    h, w = mask.shape
    x, y, bw, bh = bbox
    x0 = max(0, int(x - margin)); y0 = max(0, int(y - margin))
    x1 = min(w, int(x + bw + margin)); y1 = min(h, int(y + bh + margin))

    local_mask = np.zeros_like(mask, dtype=bool)
    local_mask[y0:y1, x0:x1] = True
    bg_local = local_mask & (~mask)
    ys, xs = np.where(bg_local)

    if len(xs) < n:
        ys, xs = np.where(~mask)
    if len(xs) == 0:
        return get_center_point(mask)

    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(xs), size=n, replace=False)
    return np.column_stack([xs[idx], ys[idx]]).astype(float)


def get_bbox_xyxy(bbox):
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h])


def get_corner_points(bbox):
    x, y, w, h = bbox
    return np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=float)


def build_prompts(local_mask, bbox, x_offset, y_offset):
    """
    `local_mask` is a mask decoded into a SMALL LOCAL crop (see
    decode_local_mask), not the full image. All point-based prompts
    (center, interior, negative) are computed in local coordinates
    then shifted by (x_offset, y_offset) to return them in ORIGINAL
    image pixel space, since that's what SAM's predict() expects.
    Box/corner prompts use `bbox` directly (already global) and need
    no shifting.
    """
    def shift(pts):
        p = pts.copy()
        p[:, 0] += x_offset
        p[:, 1] += y_offset
        return p

    cp = shift(get_center_point(local_mask))
    mp3 = shift(get_interior_points(local_mask, 3))
    mp5 = shift(get_interior_points(local_mask, 5))
    mp10 = shift(get_interior_points(local_mask, 10))
    corners = get_corner_points(bbox)
    bb = get_bbox_xyxy(bbox)
    neg = shift(get_negative_points(local_mask, [bbox[0] - x_offset, bbox[1] - y_offset, bbox[2], bbox[3]], n=3))

    return {
        "CP":  {"points": cp,  "labels": np.array([1]),          "box": None},
        "BB":  {"points": None, "labels": None,                  "box": bb},
        "MP3": {"points": mp3, "labels": np.ones(3, dtype=int),  "box": None},
        "MP5": {"points": mp5, "labels": np.ones(5, dtype=int),  "box": None},
        "MP10":{"points": mp10,"labels": np.ones(10, dtype=int), "box": None},
        "BPC": {"points": cp,  "labels": np.array([1]),          "box": bb},
        "CPC": {"points": np.vstack([corners, cp]), "labels": np.ones(5, dtype=int), "box": None},
        "CPN": {"points": np.vstack([cp, neg]), "labels": np.array([1, 0, 0, 0]), "box": None},
    }, neg  # return neg separately for the sanity check


CROP_MARGIN = 150  # local decode region around each object's bbox, in px.
                    # Must be >= the CPN negative-point sampling margin below.


def decode_local_mask(segmentation, bbox, img_h, img_w, margin=CROP_MARGIN):
    """
    Decodes the object's mask into a SMALL LOCAL crop instead of a full
    image-sized array. DOTA/iSAID images can be up to 20,000x20,000 px;
    allocating and populating a full-canvas array for every one of
    ~116,000 instances (needed only for local point sampling, never for
    the actual object extent far from it) was the dominant cost driving
    the ~123min/30-image smoke test result. This decodes only the region
    around the object, which is all prompt generation (center point,
    interior points, negative points, corners) ever needs.

    IMPORTANT: this shifts the polygon coordinates into local space and
    still uses pycocotools' OWN rasterizer (frPyObjects + decode) on the
    smaller canvas — NOT a custom OpenCV rasterizer. An earlier version
    of this function used cv2.fillPoly directly and was verified to
    produce systematically different (larger) pixel counts than
    pycocotools' actual rasterization rule — a real bug caught before
    being used on real data (see /tmp/verify_local_decode.py). This
    version was verified pixel-identical to full-canvas decode across
    15 random polygon test cases (see /tmp/verify_local_decode_v2.py).

    Returns (local_mask, x_offset, y_offset) — offsets convert local
    pixel coordinates back to original image coordinates by simple
    addition.

    Falls back to full-canvas pycocotools decode (slower, but correct)
    for the rare case where segmentation isn't polygon-format (e.g. an
    RLE-encoded crowd region).
    """
    x, y, w, h = bbox
    x0 = max(0, int(x - margin)); y0 = max(0, int(y - margin))
    x1 = min(img_w, int(x + w + margin)); y1 = min(img_h, int(y + h + margin))
    crop_h, crop_w = max(1, y1 - y0), max(1, x1 - x0)

    is_polygon = isinstance(segmentation, list) and len(segmentation) > 0 \
        and isinstance(segmentation[0], list)

    if is_polygon:
        shifted_seg = []
        for poly in segmentation:
            p = np.array(poly, dtype=np.float64).reshape(-1, 2)
            p[:, 0] -= x0
            p[:, 1] -= y0
            shifted_seg.append(p.flatten().tolist())
        rle = maskUtils.frPyObjects(shifted_seg, crop_h, crop_w)
        local_mask = maskUtils.decode(rle)
        if local_mask.ndim == 3:
            local_mask = local_mask[:, :, 0]
        return local_mask.astype(bool), x0, y0
    else:
        # fallback: rare RLE-encoded instance, full decode then crop
        rle = maskUtils.frPyObjects(segmentation, img_h, img_w)
        full = maskUtils.decode(rle)
        if full.ndim == 3:
            full = full[:, :, 0]
        return full[y0:y1, x0:x1].astype(bool), x0, y0


def compute_metrics_rle(pred_mask, gt_rle):
    """
    IoU/precision/recall computed via RLE-native operations
    (pycocotools maskUtils.iou/area) instead of dense full-canvas
    boolean array comparisons. This gives EXACT the same numbers as
    the dense-array version — it is a performance change only, not an
    approximation — while avoiding repeated full-image-sized numpy
    operations (done 8x per instance, once per strategy, in the
    dense-array version). Critically, this does NOT crop or truncate
    the predicted mask in any way, so it still correctly counts false
    positives anywhere in the image, including cases where a point
    prompt causes SAM to badly over-segment far from the object — the
    exact failure mode this paper's point-vs-box comparison depends on
    measuring accurately.
    """
    pred_rle = maskUtils.encode(np.asfortranarray(pred_mask.astype(np.uint8)))
    iou_val = float(maskUtils.iou([pred_rle], [gt_rle], [0])[0][0])
    pred_area = float(maskUtils.area(pred_rle))
    gt_area = float(maskUtils.area(gt_rle))
    inter = iou_val * (pred_area + gt_area) / (1.0 + iou_val) if iou_val > 0 else 0.0
    prec = inter / (pred_area + 1e-8)
    rec = inter / (gt_area + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return iou_val, prec, rec, f1


def sanity_check_negative_points(mask, neg_points):
    """Confirms negative points actually land outside the object mask."""
    h, w = mask.shape
    ok = 0
    for (x, y) in neg_points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w and not mask[yi, xi]:
            ok += 1
    return ok, len(neg_points)


# ═══════════════════════════════════════════════════════════════════
# MAIN INFERENCE LOOP
# ═══════════════════════════════════════════════════════════════════

def run(instances, coco, img_dir, smoke=False):
    print("Loading SAM ViT-H...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    print("SAM loaded.\n")

    from collections import defaultdict
    by_image = defaultdict(list)
    for inst in instances:
        by_image[inst["image_id"]].append(inst)
    image_ids = list(by_image.keys())

    if smoke:
        image_ids = image_ids[:30]
        print(f"[SMOKE TEST] limited to {len(image_ids)} images\n")

    # CHECKPOINTING IS NOW PER-INSTANCE (ann_id), not per-image. Some
    # images contain thousands of instances (Small_Vehicle is heavily
    # concentrated in a subset of images) — checkpointing only after a
    # full image finishes meant a long, dense image could be interrupted
    # with ALL of its work lost, and gave zero visibility into progress
    # during that image. This tracks completed ann_ids directly, so any
    # interrupt loses at most the single in-flight instance, and resume
    # correctly skips already-completed instances even mid-image.
    done_ann_ids = set()
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE) as f:
            done_ann_ids = set(int(l.strip()) for l in f if l.strip())
        print(f"Resuming: {len(done_ann_ids)} instances already completed.\n")

    # Filter each image's instance list down to what's NOT yet done,
    # and drop images that are already fully complete.
    remaining_image_ids = []
    for iid in image_ids:
        by_image[iid] = [inst for inst in by_image[iid] if inst["ann_id"] not in done_ann_ids]
        if len(by_image[iid]) > 0:
            remaining_image_ids.append(iid)
    remaining = remaining_image_ids
    total_remaining_instances = sum(len(by_image[iid]) for iid in remaining)
    print(f"Images with remaining work: {len(remaining)}  "
          f"(instances remaining: {total_remaining_instances})\n")

    write_header = not os.path.exists(RESULTS_FILE)
    csv_f = open(RESULTS_FILE, "a", newline="")

    neg_check_done = False
    neg_ok_total, neg_total = 0, 0
    images_since_cleanup = 0

    t_start = time.time()
    for img_id in tqdm(remaining, desc="Processing images", ncols=70):
        insts = by_image[img_id]
        fname = insts[0]["file_name"]
        img_path = os.path.join(img_dir, fname)

        # Diagnostic: show instance count for this image BEFORE
        # processing it, so an unusually dense image is visible in
        # real time instead of looking like a silent stall.
        n_insts = len(insts)
        if n_insts > 100:
            tqdm.write(f"  [{fname}] {n_insts} instances — this image will take a while ({n_insts*8} SAM calls)")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            tqdm.write(f"  WARNING: could not load {fname} — marking its {n_insts} instances as skipped")
            with open(RESUME_FILE, "a") as rf:
                for inst in insts:
                    rf.write(f"{inst['ann_id']}\n")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = insts[0]["img_h"], insts[0]["img_w"]

        # >>> pass the ORIGINAL image directly — SAM handles resize/pad <<<
        predictor.set_image(img_rgb)

        rows = []
        for inst_idx, inst in enumerate(insts):
            if n_insts > 100 and inst_idx > 0 and inst_idx % 50 == 0:
                tqdm.write(f"    ... {inst_idx}/{n_insts} instances done in this image")
            ann = coco.loadAnns([inst["ann_id"]])[0]

            # Local decode for prompt generation — small crop, not full canvas
            local_mask, x_off, y_off = decode_local_mask(
                ann["segmentation"], inst["bbox"], H, W)

            # Separate RLE (compact, not decoded to dense) for metrics —
            # this matches the FULL image canvas since SAM's predicted
            # masks come back at full resolution and must be compared
            # against a GT representation at the same canvas size.
            gt_rle = maskUtils.frPyObjects(ann["segmentation"], H, W)
            if isinstance(gt_rle, list):
                gt_rle = maskUtils.merge(gt_rle)

            prompts, neg_pts = build_prompts(local_mask, inst["bbox"], x_off, y_off)

            if not neg_check_done:
                ok, tot = sanity_check_negative_points(local_mask, neg_pts - np.array([x_off, y_off]))
                neg_ok_total += ok
                neg_total += tot

            for strat in STRATEGIES:
                p = prompts[strat]
                t0 = time.perf_counter()
                try:
                    masks, scores, _ = predictor.predict(
                        point_coords=p["points"], point_labels=p["labels"],
                        box=p["box"], multimask_output=True)
                    t1 = time.perf_counter()
                    best = np.argmax(scores)
                    # RLE-native metric computation — no full-canvas dense
                    # array comparison needed (see compute_metrics_rle docstring)
                    iou, prec, rec, f1 = compute_metrics_rle(masks[best], gt_rle)
                    conf = float(scores[best])
                    ms = (t1 - t0) * 1000
                    err = ""
                except Exception as e:
                    iou = prec = rec = f1 = conf = ms = np.nan
                    err = str(e)[:80]

                rows.append({
                    "ann_id": inst["ann_id"], "image_id": img_id,
                    "category": inst["category"], "area_px": inst["area_px"],
                    "stratum": inst["stratum"], "strategy": strat,
                    "iou": iou, "precision": prec, "recall": rec, "f1": f1,
                    "confidence": conf, "time_ms": ms, "error": err,
                })

            # Write and checkpoint THIS INSTANCE immediately (all 8
            # strategy rows for it), rather than waiting for the whole
            # image to finish. This is what makes checkpointing safe
            # for images with thousands of instances — an interrupt
            # loses at most the one instance currently in flight.
            inst_rows = rows[-len(STRATEGIES):]
            pd.DataFrame(inst_rows).to_csv(csv_f, header=write_header, index=False)
            write_header = False
            csv_f.flush()
            with open(RESUME_FILE, "a") as rf:
                rf.write(f"{inst['ann_id']}\n")

        if not neg_check_done and neg_total >= 50:
            pct = 100 * neg_ok_total / neg_total
            print(f"\n[CPN sanity check] {neg_ok_total}/{neg_total} ({pct:.1f}%) "
                  f"negative points confirmed in background.")
            if pct < 95:
                print("  WARNING: unexpectedly low — check get_negative_points() logic.")
            neg_check_done = True

        # Explicitly release this image's large arrays before the next
        # iteration, and periodically clear PyTorch's CUDA memory cache.
        # DOTA images vary enormously in size (800x800 to 20,000x20,000
        # px); without this, memory fragmentation accumulates over a
        # long run and processing speed degrades severely over many
        # hours (observed: ~23s/image at the start of a run degrading
        # to ~900s/image after 16 hours on an earlier run). This is a
        # standard mitigation for long-running variable-input-size
        # PyTorch workloads and has no effect on computed results.
        del img_bgr, img_rgb
        images_since_cleanup += 1
        if images_since_cleanup >= CLEANUP_INTERVAL:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            images_since_cleanup = 0

    csv_f.close()
    elapsed = time.time() - t_start
    print(f"\nDone. Elapsed: {elapsed/60:.1f} min for {len(remaining)} images.")
    print(f"Results appended to: {RESULTS_FILE}")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    set_determinism(SEED)
    write_environment_manifest()

    img_dir = preflight()
    instances, coco = load_all_instances(img_dir)
    if len(instances) == 0:
        print("ERROR: 0 valid instances loaded.")
        sys.exit(1)

    run(instances, coco, img_dir, smoke=args.smoke)
