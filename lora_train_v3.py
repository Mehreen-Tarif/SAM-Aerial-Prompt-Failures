r"""
lora_train_v3.py -- LoRA fine-tuning of SAM ViT-H (box-only adapter).

Fixes relative to lora_sam_train.py / lora_stratified.py:
  1. PREPROCESSING: images go through SAM's own ResizeLongestSide + sam.preprocess on 0-255
     values -- byte-for-byte the SamPredictor path used at evaluation (no extra scaling of the pixel values).
     A start-up SELF-TEST checks this: training-path features must equal SamPredictor features.
  2. VALIDATION never touches iSAID val: a fixed, seeded 5% of TRAINING images is held out.
     The same validation instances are used every epoch.
  3. Validation IoU is scored at NATIVE resolution with the v4 metric (not at 1024 px).
  4. CHECKPOINT RULE fixed before training (written to config.json): the epoch with the highest
     macro-over-strata mIoU (mean of small/medium/large mIoU) on the held-out set; ties -> earlier.
  5. Input mode matches evaluation: --mode tile1024 trains on the same native-resolution tiles
     that sam_eval_v4.py evaluates on (or --mode whole for whole images).

Prompts: GT box (BB), multimask_output=True, loss on the best of the 3 candidates (SAM's own
training convention), BCE + Dice at SAM-input resolution. Only LoRA parameters train.

Usage (from the repository folder, AFTER the evaluation runs have finished -- one GPU job at a time):
    python lora_train_v3.py --mode tile1024 --smoke        # ~10 min check
    python lora_train_v3.py --mode tile1024                # full run (resumable: add --resume)
Then evaluate the chosen checkpoint with the validated pipeline:
    python sam_eval_v4.py --model sam1 --mode tile1024 --lora lora_v3\tile1024\best.pt --tag v3
"""
import os, sys, json, time, math, argparse, random
import numpy as np, cv2, torch, torch.nn.functional as F

B = os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, B); sys.path.insert(0, os.path.join(B, "scripts"))
import sam_experiment_v3 as v3
import sam_eval_v4 as v4
from lora_sam import inject_lora, lora_state, load_lora
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

TRAIN_ANN = os.path.join(v3.DATA_DIR, "data", "iSAID", "Annotations", "iSAID_train_20190823_114751.json")
TRAIN_IMG = os.path.join(v3.DATA_DIR, "data", "iSAID", "train", "images")

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="tile1024")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--windows-per-epoch", type=int, default=3000)
    ap.add_argument("--max-inst", type=int, default=24, help="instances per window per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-inst", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: a.epochs, a.windows_per_epoch, a.val_inst = 1, 30, 150
    return a

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_rgb(path):
    im = cv2.imread(path)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def valid_anns(coco, img_id, H, W):
    out = []
    for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0 or not a.get("segmentation") or not isinstance(a["segmentation"], list): continue
        g, _, _ = v4.gt_local(a["segmentation"], a["bbox"], H, W)
        if g.any(): out.append(a)
    return out

def gt_at_input(seg, wx, wy, s, ih, iw):
    """GT polygon(s) shifted into the window and scaled to SAM's input size, rasterised by pycocotools."""
    polys = [((np.array(p, np.float64).reshape(-1, 2) - [wx, wy]) * s).flatten().tolist() for p in seg]
    r = maskUtils.frPyObjects(polys, ih, iw)
    r = maskUtils.merge(r) if isinstance(r, list) else r
    return maskUtils.decode(r).astype(np.float32)

def encode(sam, tf, img_rgb, dev):
    """Exactly SamPredictor.set_image: ResizeLongestSide -> uint8 tensor (0-255) -> sam.preprocess."""
    inp = tf.apply_image(img_rgb)
    x = torch.as_tensor(inp, device=dev).permute(2, 0, 1).contiguous()[None, :, :, :]
    return sam.image_encoder(sam.preprocess(x)), inp.shape[:2]

def dice_loss(logits, target):
    p = torch.sigmoid(logits).flatten(-2); t = target.flatten(-2)
    return 1 - (2 * (p * t).sum(-1) + 1) / (p.sum(-1) + t.sum(-1) + 1)

def evaluate(sam, val_items, coco_tr, mode, log):
    """Native-resolution BB evaluation on the fixed held-out TRAIN instances (v4 metric)."""
    pred = SamPredictor(sam); res = {"small": [], "medium": [], "large": []}
    with torch.no_grad():
        for img_id, path, H, W, anns in val_items:
            img = load_rgb(path)
            if img is None: continue
            wins, amap = v4.assign_windows(anns, H, W, mode)
            for key, (wx, wy, ww, wh, kind) in wins.items():
                members = [a for a in anns if amap[a["id"]] == key]
                if not members: continue
                pred.set_image(np.ascontiguousarray(img[wy:wy + wh, wx:wx + ww]))
                for a in members:
                    x, y, w, h = a["bbox"]
                    box = np.array([x - wx, y - wy, x + w - wx, y + h - wy], float)
                    masks, scores, _ = pred.predict(box=box, multimask_output=True)
                    g, gx0, gy0 = v4.gt_local(a["segmentation"], a["bbox"], H, W)
                    iou = v4.fast_metrics(masks[int(np.argmax(scores))], g, gx0 - wx, gy0 - wy, float(g.sum()))[0]
                    res[v3.stratum_of(int(a.get("area", 0)))].append(iou)
    out = {}
    for z, v in res.items():
        v = np.array(v); out[z] = dict(n=int(len(v)), miou=float(v.mean()) if len(v) else float("nan"),
                                       fail=float(100 * (v < .5).mean()) if len(v) else float("nan"))
    allv = np.concatenate([np.array(v) for v in res.values() if len(v)])
    out["all"] = dict(n=int(len(allv)), miou=float(allv.mean()), fail=float(100 * (allv < .5).mean()))
    out["macro_strata_miou"] = float(np.nanmean([out[z]["miou"] for z in ["small", "medium", "large"]]))
    return out

def main():
    a = parse(); seed_all(a.seed)
    out_dir = os.path.join(B, "lora_v3", a.mode + ("_smoke" if a.smoke else "")); os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "train_log.txt"), "a", encoding="utf-8")
    def log(s): print(s); logf.write(s + "\n"); logf.flush()
    dev = v3.DEVICE
    log(f"== {time.strftime('%Y-%m-%d %H:%M:%S')}  lora_train_v3  {vars(a)}")

    # ---- data: fixed seeded split of TRAINING images into train / held-out validation
    coco = COCO(TRAIN_ANN)
    ids = sorted(coco.getImgIds()); rng = random.Random(a.seed); rng.shuffle(ids)
    n_val = max(3, int(round(a.val_frac * len(ids))))
    val_ids, tr_ids = sorted(ids[:n_val]), sorted(ids[n_val:])
    with open(os.path.join(out_dir, "heldout_train_images.txt"), "w") as f:
        f.write("\n".join(coco.loadImgs([i])[0]["file_name"] for i in val_ids))
    log(f"train images {len(tr_ids)}, held-out validation images {len(val_ids)} (from the TRAIN split; iSAID val untouched)")

    def img_meta(i):
        info = coco.loadImgs([i])[0]; p = os.path.join(TRAIN_IMG, info["file_name"])
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None: return None
        H, W = im.shape[:2]; return p, H, W
    # round-robin over ALL held-out images, so no single dense scene dominates validation
    pools, vrng = [], random.Random(a.seed + 1)
    for i in val_ids:
        m = img_meta(i)
        if m is None: continue
        anns = valid_anns(coco, i, m[1], m[2]); vrng.shuffle(anns)
        if anns: pools.append((i, m, anns))
    chosen = {i: [] for i, _, _ in pools}
    n_v, k = 0, 0
    while n_v < a.val_inst and any(k < len(an) for _, _, an in pools):
        for i, m, an in pools:
            if k < len(an) and n_v < a.val_inst:
                chosen[i].append(an[k]); n_v += 1
        k += 1
    val_items = [(i, m[0], m[1], m[2], chosen[i]) for i, m, _ in pools if chosen[i]]
    log(f"fixed validation set: {n_v} instances from {len(val_items)} held-out images (same every epoch)")

    cfg = dict(vars(a), selection_rule="argmax over epochs of macro-over-strata mIoU (mean of small/medium/large "
               "mIoU, BB prompt, native resolution) on the fixed held-out TRAIN instances; ties -> earlier epoch",
               train_images=len(tr_ids), heldout_images=len(val_ids), val_instances=n_v,
               written=time.strftime('%Y-%m-%d %H:%M:%S'))
    cfg_p = os.path.join(out_dir, "config.json")
    if not os.path.exists(cfg_p): json.dump(cfg, open(cfg_p, "w"), indent=2)

    # ---- model
    sam = sam_model_registry["vit_h"](checkpoint=v3.SAM_CKPT).to(dev); sam.eval()
    n_tr = inject_lora(sam, a.rank, a.alpha); sam.to(dev)
    log(f"LoRA injected: {n_tr:,} trainable parameters")
    tf = ResizeLongestSide(sam.image_encoder.img_size)

    # ---- SELF-TEST: training preprocessing must equal SamPredictor exactly (catches any scaling mismatch)
    m = img_meta(val_items[0][0]); img = load_rgb(m[0])[:1024, :1024]
    with torch.no_grad():
        f_train, _ = encode(sam, tf, img, dev)
        p = SamPredictor(sam); p.set_image(img); f_pred = p.features
    diff = (f_train - f_pred).abs().max().item()
    log(f"SELF-TEST preprocessing: max |train-path features - SamPredictor features| = {diff:.2e}")
    if diff > 1e-4: log("!! SELF-TEST FAILED -- stopping. See the log."); return
    log("SELF-TEST PASS")

    params = [p for p in sam.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    total = a.epochs * a.windows_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    hist_p = os.path.join(out_dir, "history.json")
    hist, start_ep = (json.load(open(hist_p)) if os.path.exists(hist_p) else []), 0
    if hist and not a.resume:
        log(f"!! {out_dir} already has a run. Use --resume to continue it, or delete the folder to start over."); return
    if a.resume:
        cks = sorted(f for f in os.listdir(out_dir) if f.startswith("epoch_") and f.endswith(".pt"))
        if cks:
            ck = load_lora(sam, os.path.join(out_dir, cks[-1]))
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); start_ep = ck["epoch"]
            log(f"resumed after epoch {start_ep}")
    if not hist:
        log("epoch 0 (untrained = frozen SAM) validation ...")
        r0 = evaluate(sam, val_items, coco, a.mode, log); hist.append(dict(epoch=0, **r0))
        json.dump(hist, open(hist_p, "w"), indent=2); log(f"  epoch 0: {json.dumps(r0)}")

    trng = random.Random(a.seed + 100)
    for ep in range(start_ep + 1, a.epochs + 1):
        t0, steps, loss_sum = time.time(), 0, 0.0
        order = tr_ids[:]; random.Random(a.seed + ep).shuffle(order)
        for i in order:
            if steps >= a.windows_per_epoch: break
            m = img_meta(i)
            if m is None: continue
            path, H, W = m; anns = valid_anns(coco, i, H, W)
            if not anns: continue
            img = load_rgb(path)
            wins, amap = v4.assign_windows(anns, H, W, a.mode)
            keys = list(wins); trng.shuffle(keys)
            for key in keys:
                if steps >= a.windows_per_epoch: break
                wx, wy, ww, wh, kind = wins[key]
                members = [x for x in anns if amap[x["id"]] == key]
                if not members: continue
                trng.shuffle(members); members = members[:a.max_inst]
                emb, (ih, iw) = encode(sam, tf, np.ascontiguousarray(img[wy:wy + wh, wx:wx + ww]), dev)
                s = ih / wh
                boxes = np.array([[x["bbox"][0] - wx, x["bbox"][1] - wy, x["bbox"][0] + x["bbox"][2] - wx,
                                   x["bbox"][1] + x["bbox"][3] - wy] for x in members], float)
                bt = torch.as_tensor(tf.apply_boxes(boxes, (wh, ww)), dtype=torch.float, device=dev)
                with torch.no_grad():
                    sparse, dense = sam.prompt_encoder(points=None, boxes=bt, masks=None)
                low, _ = sam.mask_decoder(image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
                                          sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                          multimask_output=True)
                logits = sam.postprocess_masks(low, (ih, iw), (ih, iw))          # (N, 3, ih, iw)
                gt = torch.as_tensor(np.stack([gt_at_input(x["segmentation"], wx, wy, s, ih, iw) for x in members]),
                                     device=dev)[:, None].expand_as(logits)
                per = F.binary_cross_entropy_with_logits(logits, gt, reduction="none").mean((-1, -2)) + dice_loss(logits, gt)
                loss = per.min(1).values.mean()                                   # best of 3 candidates
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                steps += 1; loss_sum += loss.item()
                if steps % 100 == 0 or a.smoke:
                    log(f"  ep {ep} step {steps}/{a.windows_per_epoch}  loss {loss_sum/steps:.4f}  "
                        f"lr {sched.get_last_lr()[0]:.2e}  {(time.time()-t0)/steps:.2f}s/step")
            del img
        ck = dict(lora=lora_state(sam), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, config=cfg)
        torch.save(ck, os.path.join(out_dir, f"epoch_{ep:03d}.pt"))
        r = evaluate(sam, val_items, coco, a.mode, log); hist.append(dict(epoch=ep, train_loss=loss_sum / max(1, steps), **r))
        json.dump(hist, open(hist_p, "w"), indent=2)
        log(f"epoch {ep}: loss {loss_sum/max(1,steps):.4f}  val {json.dumps(r)}  ({(time.time()-t0)/60:.1f} min)")
        torch.cuda.empty_cache()

    best = max([h for h in hist if h["epoch"] > 0], key=lambda h: (h["macro_strata_miou"], -h["epoch"]))
    import shutil
    shutil.copy(os.path.join(out_dir, f"epoch_{best['epoch']:03d}.pt"), os.path.join(out_dir, "best.pt"))
    log(f"== selection rule -> epoch {best['epoch']} (macro-strata mIoU {best['macro_strata_miou']:.4f}; "
        f"epoch 0 baseline {hist[0]['macro_strata_miou']:.4f}) -> best.pt")

if __name__ == "__main__":
    main()
