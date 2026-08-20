r"""
DATASET INTEGRITY AUDIT — run this BEFORE the SAM experiment
================================================================
Checks, without touching SAM or the GPU at all:
  1. Annotation file parses correctly
  2. Category names present, compared against the canonical iSAID-15 list
  3. How many referenced images actually exist on disk (and which don't)
  4. How many annotations decode to a valid, non-empty mask
  5. Size-stratum distribution (small/medium/large) across valid instances
  6. SAM checkpoint file size sanity check (catches a corrupted/partial
     download, which silently produces garbage predictions otherwise)

Run:
    python audit_dataset.py

Output: prints a report AND writes results/audit_report.txt
"""

import os
import sys
from collections import Counter, defaultdict

from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# ── same config as sam_experiment_v2.py — keep these in sync ──────────
DATA_DIR = r"D:\iSAID_Data"
ANN_FILE = os.path.join(DATA_DIR, "data", "iSAID", "Annotations",
                         "iSAID_val_20190823_114742.json")
IMG_DIR_CANDIDATES = [
    os.path.join(DATA_DIR, "data", "iSAID", "val", "images"),
    os.path.join(DATA_DIR, "data", "iSAID", "val"),
]
SAM_CKPT = os.path.join(DATA_DIR, "sam_vit_h_4b8939.pth")

PROJECT_DIR = r"C:\Users\Lenovo\Desktop\SAM_Revision"
OUT_DIR = os.path.join(PROJECT_DIR, "results")

# Canonical iSAID 15 categories (official paper naming) — used only to
# flag naming mismatches, not to filter anything.
CANONICAL_CATS = {
    "ship", "storage_tank", "baseball_diamond", "tennis_court",
    "basketball_court", "Ground_Track_Field", "Bridge", "Large_Vehicle",
    "Small_Vehicle", "Helicopter", "Swimming_pool", "Roundabout",
    "Soccer_ball_field", "plane", "Harbor",
}

SMALL_MAX = 32 * 32
MEDIUM_MAX = 96 * 96


def stratum_of(area):
    if area < SMALL_MAX:
        return "small"
    elif area < MEDIUM_MAX:
        return "medium"
    return "large"


def resolve_img_dir():
    for cand in IMG_DIR_CANDIDATES:
        if os.path.isdir(cand):
            n = len([f for f in os.listdir(cand)
                      if f.lower().endswith((".png", ".jpg", ".tif"))])
            if n > 0:
                return cand, n
    return None, 0


def main():
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("  DATASET INTEGRITY AUDIT")
    log("=" * 70)

    # ── 1. Annotation file ──────────────────────────────────────────
    if not os.path.exists(ANN_FILE):
        log(f"FATAL: annotation file not found at {ANN_FILE}")
        sys.exit(1)
    log(f"\n[1] Annotation file: {ANN_FILE}")
    try:
        coco = COCO(ANN_FILE)
    except Exception as e:
        log(f"FATAL: could not parse annotation file: {e}")
        sys.exit(1)

    n_images_ann = len(coco.getImgIds())
    n_anns = len(coco.getAnnIds())
    log(f"    Images referenced in annotation file: {n_images_ann}")
    log(f"    Total annotations: {n_anns}")

    # ── 2. Category names ────────────────────────────────────────────
    cats = coco.loadCats(coco.getCatIds())
    cat_names = {c["name"] for c in cats}
    log(f"\n[2] Categories found ({len(cats)}):")
    for c in sorted(cat_names):
        flag = "" if c in CANONICAL_CATS else "  <-- NOT in canonical iSAID-15 list, check spelling"
        log(f"    - {c}{flag}")
    missing_canonical = CANONICAL_CATS - cat_names
    if missing_canonical:
        log(f"\n    WARNING: canonical categories not found in file: {sorted(missing_canonical)}")
        log(f"    (This may just mean the annotation file uses different spelling —")
        log(f"     compare against the flagged names above.)")

    # ── 3. Image folder match ────────────────────────────────────────
    img_dir, n_on_disk = resolve_img_dir()
    log(f"\n[3] Image folder resolution:")
    if img_dir is None:
        log(f"    FATAL: no candidate image folder contains PNG/JPG files.")
        log(f"    Checked: {IMG_DIR_CANDIDATES}")
        sys.exit(1)
    log(f"    Using: {img_dir}")
    log(f"    Files on disk: {n_on_disk}")

    available = set(os.listdir(img_dir))
    all_img_ids = coco.getImgIds()
    matched, missing = 0, []
    for iid in all_img_ids:
        fname = coco.loadImgs([iid])[0]["file_name"]
        if fname in available:
            matched += 1
        else:
            missing.append(fname)
    log(f"    Matched to annotation references: {matched} / {n_images_ann}")
    if missing:
        log(f"    Missing: {len(missing)} (showing first 10): {missing[:10]}")

    # ── 4. Validity + stratum distribution ──────────────────────────
    # NOTE: we use the precomputed COCO 'area' field rather than fully
    # decoding every mask. DOTA/iSAID images can be up to 20,000x20,000
    # px; decoding a full-image-sized array for every one of ~116,000
    # annotations just to measure area was the cause of this audit
    # taking an extremely long time with no progress feedback. The
    # 'area' field is already computed and stored in the annotation
    # JSON, so we can validate size/stratum instantly without decoding.
    # We separately decode a small RANDOM SAMPLE to confirm mask decode
    # itself works correctly (catches corrupt segmentation data) without
    # paying the full cost across all annotations.
    log(f"\n[4] Validating all {n_anns} annotations (using precomputed area field)...")
    skip = Counter()
    cat_counts = Counter()
    strat_counts = Counter()
    cat_strat = defaultdict(Counter)
    valid = 0

    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    all_ann_ids = coco.getAnnIds()

    for aid in tqdm(all_ann_ids, desc="    Validating", ncols=70):
        ann = coco.loadAnns([aid])[0]
        img_info = coco.loadImgs([ann["image_id"]])[0]
        if img_info["file_name"] not in available:
            skip["no_image"] += 1
            continue
        seg = ann.get("segmentation")
        if not seg:
            skip["no_seg"] += 1
            continue

        area = ann.get("area", 0)
        if area <= 0:
            skip["zero_area"] += 1
            continue

        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            skip["bad_bbox"] += 1
            continue

        valid += 1
        cname = cat_id_to_name.get(ann["category_id"], "unknown")
        strat = stratum_of(int(area))
        cat_counts[cname] += 1
        strat_counts[strat] += 1
        cat_strat[cname][strat] += 1

    log(f"\n    Valid instances: {valid} / {n_anns}")
    log(f"    Skipped breakdown: {dict(skip)}")

    # ── 4b. Decode sanity check on a random sample (catches corrupt data) ──
    log(f"\n[4b] Decode sanity check on a random sample of 300 annotations...")
    import random as _random
    _random.seed(42)
    sample_ids = _random.sample(all_ann_ids, min(300, len(all_ann_ids)))
    decode_ok, decode_fail = 0, 0
    for aid in tqdm(sample_ids, desc="    Decode check", ncols=70):
        ann = coco.loadAnns([aid])[0]
        img_info = coco.loadImgs([ann["image_id"]])[0]
        try:
            rle = maskUtils.frPyObjects(ann["segmentation"], img_info["height"], img_info["width"])
            mask = maskUtils.decode(rle)
            decode_ok += 1
        except Exception:
            decode_fail += 1
    log(f"    Decode OK: {decode_ok} / {len(sample_ids)}  (fail: {decode_fail})")
    if decode_fail > 0:
        log(f"    WARNING: {decode_fail} decode failures in sample — investigate before full run.")

    log(f"\n    Valid instances: {valid} / {n_anns}")
    log(f"    Skipped breakdown: {dict(skip)}")

    log(f"\n[5] Per-category counts:")
    for c in sorted(cat_counts):
        log(f"    {c:<22} {cat_counts[c]:>6}")

    log(f"\n[6] Size-stratum distribution (COCO thresholds: small<1024px2, medium<9216px2, large>=9216px2):")
    total = sum(strat_counts.values())
    for s in ["small", "medium", "large"]:
        n = strat_counts.get(s, 0)
        pct = 100 * n / total if total else 0
        log(f"    {s:<10} {n:>6}  ({pct:.1f}%)")

    log(f"\n[7] Per-category x stratum (flags categories missing a whole stratum):")
    for c in sorted(cat_strat):
        row = cat_strat[c]
        flags = [s for s in ["small", "medium", "large"] if row.get(s, 0) == 0]
        flag_str = f"   <-- MISSING: {flags}" if flags else ""
        log(f"    {c:<22} small={row.get('small',0):>5}  medium={row.get('medium',0):>5}  large={row.get('large',0):>5}{flag_str}")

    # ── 8. SAM checkpoint sanity check ───────────────────────────────
    log(f"\n[8] SAM checkpoint:")
    if not os.path.exists(SAM_CKPT):
        log(f"    FATAL: checkpoint not found at {SAM_CKPT}")
    else:
        size_mb = os.path.getsize(SAM_CKPT) / (1024 * 1024)
        # official vit_h checkpoint is ~2564 MB
        log(f"    File: {SAM_CKPT}")
        log(f"    Size: {size_mb:.1f} MB")
        if size_mb < 2400:
            log(f"    WARNING: expected ~2564 MB for ViT-H. This looks corrupted or partially downloaded.")
        else:
            log(f"    OK — size matches expected ViT-H checkpoint.")

    log("\n" + "=" * 70)
    log("  AUDIT COMPLETE")
    log("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "audit_report.txt"), "w") as f:
        f.write("\n".join(lines))
    log(f"\nFull report written to: {os.path.join(OUT_DIR, 'audit_report.txt')}")


if __name__ == "__main__":
    main()
