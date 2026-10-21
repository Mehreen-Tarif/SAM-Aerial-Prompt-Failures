r"""
harbor_check.py -- why do the iSAID-trained adapters fail on NWPU harbours?  (CPU only, safe to run during training)
  Part 1: NWPU harbour IoU / precision / recall for frozen, v3, v4 (box and centroid prompts).
          precision high + recall low  -> the adapters segment only PART of what NWPU labels as harbour.
  Part 2: draws 3 iSAID val images with harbour (yellow) and ship (cyan) outlines, to compare the label definitions.
Run from the repository folder:   python harbor_check.py
Output: harbor_check\report.txt and harbor_check\isaid_*.png
"""
import os, json
import pandas as pd
from PIL import Image, ImageDraw

BASE = os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "harbor_check"); os.makedirs(OUT, exist_ok=True)
REP = open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8")
def say(s=""): print(s); REP.write(str(s) + "\n"); REP.flush()

# ---------------- Part 1: NWPU harbour precision / recall
say("Part 1. NWPU harbour (n=239, 27 images): medians and failure rate")
for run in ["sam1_tile1024", "sam1lora-v3_tile1024", "sam1lora-v4_tile1024"]:
    p = os.path.join(BASE, "results_ext_nwpu", run, "results.csv")
    if not os.path.exists(p): say(f"  missing {p}"); continue
    D = pd.read_csv(p, low_memory=False)
    for s in ["BB", "CP"]:
        X = D[(D.category == "harbor") & (D.strategy == s)]
        ratio = (X.recall / X.precision.clip(lower=1e-6))  # = predicted area / true area
        say(f"  {run:22s} {s:3s} n={len(X):3d}  fail {100 * (X.iou < 0.5).mean():5.1f}%  IoU {X.iou.median():.3f}  "
            f"precision {X.precision.median():.3f}  recall {X.recall.median():.3f}  pred/true area {ratio.median():.2f}")
    X = D[(D.category != "harbor") & (D.strategy == "BB")]
    say(f"  {run:22s} BB  all other classes: fail {100 * (X.iou < 0.5).mean():.1f}%")

# ---------------- Part 2: iSAID harbour vs ship labels
say("\nPart 2. iSAID val images with many harbours (yellow = harbour, cyan = ship)")
ROOT = os.path.join(os.environ.get("ISAID_ROOT", r"D:\iSAID_Data"), "data", "iSAID")
d = json.load(open(os.path.join(ROOT, "Annotations", "iSAID_val_20190823_114742.json")))  # original release (has Harbor)
cat = {c["id"]: c["name"].lower() for c in d["categories"]}
img = {i["id"]: i for i in d["images"]}
harb = [k for k, v in cat.items() if "harbor" in v or "harbour" in v]
ship = [k for k, v in cat.items() if v == "ship"]
cnt = {}
for a in d["annotations"]:
    if a["category_id"] in harb: cnt[a["image_id"]] = cnt.get(a["image_id"], 0) + 1
say(f"  categories: {sorted(cat.values())}")
say(f"  harbour ids {harb}, ship ids {ship}; images with harbours: {len(cnt)}, max per image: {max(cnt.values()) if cnt else 0}")
def size(i): return img[i].get("width", 0) * img[i].get("height", 0)
cands = sorted([i for i, n in cnt.items() if n >= 3], key=size)[:3] or sorted(cnt, key=lambda i: -cnt[i])[:3]
files = {}
for r, _, fs in os.walk(os.path.join(ROOT, "val", "images")):
    for f in fs: files[f] = os.path.join(r, f)
for i in cands:
    fn = img[i]["file_name"]
    if fn not in files: say(f"  image file not found: {fn}"); continue
    I = Image.open(files[fn]).convert("RGB"); dr = ImageDraw.Draw(I)
    for a in d["annotations"]:
        if a["image_id"] != i or a["category_id"] not in harb + ship: continue
        col = (255, 255, 0) if a["category_id"] in harb else (0, 220, 255)
        for poly in a["segmentation"] if isinstance(a["segmentation"], list) else []:
            if len(poly) >= 6: dr.line(poly + poly[:2], fill=col, width=3)
    sc = 1600 / max(I.size)
    if sc < 1: I = I.resize((int(I.width * sc), int(I.height * sc)))
    o = os.path.join(OUT, f"isaid_{os.path.splitext(fn)[0]}.png"); I.save(o)
    say(f"  {fn}: {cnt[i]} harbours -> {o}")
say("DONE")
