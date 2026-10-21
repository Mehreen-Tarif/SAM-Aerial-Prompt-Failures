r"""
lora_train_v4.py -- PROMPT-MIXED LoRA fine-tuning of SAM ViT-H with a re-trained confidence head.

Why (finding from lora_train_v3, box-only training):
  box-only LoRA cut BB failure 39.5% -> 9.3% on iSAID val, but SAM's own mask choice for POINT prompts got
  much worse (selection cost CP 8.9 -> 24.1 pp, CPN 8.9 -> 32.9 pp) although the best-of-3 masks improved
  for every prompt. The frozen IoU-prediction (confidence) head no longer ranks the candidates correctly
  on the adapted features, and only boxes were seen in training.

Changes relative to v3 (everything else -- preprocessing, self-test, held-out split, tiles, schedule --
is identical, so v3 and v4 differ only in what is being tested):
  1. PROMPTS: each training step draws ONE prompt type for its window, uniformly from
     TRAIN_STRATS = BB, BPC, CP, CPN, MP3, MP5, MP10, built by sam_eval_v4.build_all_prompts -- the exact
     prompt code used at evaluation. EXT and CPC are NEVER trained on: they test transfer to unseen prompts.
  2. CONFIDENCE HEAD: the mask decoder's IoU-prediction head (132,612 params) trains together with LoRA,
     supervised with MSE to the true IoU of each of the 3 candidates (SAM's own training objective).
  3. SELECTION RULE (fixed before training, written to config.json): epoch with the highest mean, over
     BB / CP / CPN / MP5, of macro-over-strata mIoU, scored with SAM's OWN mask choice at native
     resolution on the held-out TRAIN instances. EXT is logged for monitoring but not used for selection.

Usage (from the repository folder; needs the updated lora_sam.py next to it):
    python lora_train_v4.py --mode tile1024 --smoke      # ~10-15 min check
    python lora_train_v4.py --mode tile1024              # full run (resumable: add --resume)
Evaluate with the unchanged pipeline (the confidence head is detected in the checkpoint automatically):
    python sam_eval_v4.py --model sam1 --mode tile1024 --lora lora_v4\tile1024\best.pt --tag v4
"""
import os, sys, json, time, math, argparse, random, shutil
from collections import Counter
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
TRAIN_STRATS = ["BB", "BPC", "CP", "CPN", "MP3", "MP5", "MP10"]
SEL_STRATS = ["BB", "CP", "CPN", "MP5"]          # used by the selection rule
VAL_STRATS = SEL_STRATS + ["EXT"]                 # EXT: unseen prompt, monitored only
STRATA = ["small", "medium", "large"]


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
    ap.add_argument("--iou-w", type=float, default=1.0, help="weight of the confidence-head MSE loss")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-inst", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", default="lora+head", choices=["lora+head", "lora", "head", "decoder"],
                    help="what is fine-tuned: LoRA + confidence head (default), LoRA only, confidence head only, "
                         "or the whole mask decoder without LoRA (standard adaptation baseline)")
    ap.add_argument("--run-seed", type=int, default=None, help="seed for init / sampling order (default = --seed); "
                    "the train/validation split always uses --seed, so seeds differ only in training randomness")
    ap.add_argument("--out-name", default="lora_v4", help="output folder name under the repository folder")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: a.epochs, a.windows_per_epoch, a.val_inst = 1, 35, 150
    if a.run_seed is None: a.run_seed = a.seed
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


def finite(p):
    return p["points"] is None or np.isfinite(p["points"]).all()


def window_prompts(members, strat, H, W, wx, wy):
    """Prompts of one type for the instances of a window, in window coordinates (same code as evaluation).
    Drops instances whose prompt is undefined (empty first polygon -> NaN) and keeps one point count."""
    keep, plist = [], []
    for x in members:
        g, gx0, gy0 = v4.gt_local(x["segmentation"], x["bbox"], H, W)
        try:
            p = v4.shift_prompt(v4.build_all_prompts(x, H, W, g, gx0, gy0)[strat], wx, wy)
        except Exception:
            continue
        if finite(p): keep.append(x); plist.append(p)
    if plist and plist[0]["points"] is not None:
        k = Counter(len(p["points"]) for p in plist).most_common(1)[0][0]
        sel = [i for i, p in enumerate(plist) if len(p["points"]) == k]
        keep, plist = [keep[i] for i in sel], [plist[i] for i in sel]
    return keep, plist


def evaluate(sam, val_items, mode):
    """Native-resolution evaluation with SAM's OWN mask choice, several prompt types (v4 metric)."""
    pred = SamPredictor(sam); res = {s: {z: [] for z in STRATA} for s in VAL_STRATS}
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
                    g, gx0, gy0 = v4.gt_local(a["segmentation"], a["bbox"], H, W)
                    z = v3.stratum_of(int(a.get("area", 0)))
                    try:
                        P = v4.build_all_prompts(a, H, W, g, gx0, gy0)
                    except Exception:
                        P = None                                # same rule as evaluation: failed prompt scores 0
                    for s in VAL_STRATS:
                        p = v4.shift_prompt(P[s], wx, wy) if P is not None else None
                        if p is None or not finite(p):
                            iou = 0.0                       # same rule as evaluation: undefined prompt scores 0
                        else:
                            masks, scores, _ = pred.predict(point_coords=p["points"], point_labels=p["labels"],
                                                            box=p["box"], multimask_output=True)
                            iou = v4.fast_metrics(masks[int(np.argmax(scores))], g, gx0 - wx, gy0 - wy,
                                                  float(g.sum()))[0]
                        res[s][z].append(iou)
    out = {}
    for s in VAL_STRATS:
        r = {}
        for z in STRATA:
            v = np.array(res[s][z])
            r[z] = dict(n=int(len(v)), miou=float(v.mean()) if len(v) else float("nan"),
                        fail=float(100 * (v < .5).mean()) if len(v) else float("nan"))
        allv = np.concatenate([np.array(res[s][z]) for z in STRATA if len(res[s][z])])
        r["all"] = dict(miou=float(allv.mean()), fail=float(100 * (allv < .5).mean()))
        r["macro"] = float(np.nanmean([r[z]["miou"] for z in STRATA]))
        out[s] = r
    out["selection_score"] = float(np.mean([out[s]["macro"] for s in SEL_STRATS]))
    return out


def brief(r):
    return "  ".join(f"{s}: fail {r[s]['all']['fail']:.1f}% (S {r[s]['small']['fail']:.1f}) macro {r[s]['macro']:.3f}"
                     for s in VAL_STRATS) + f"  | selection score {r['selection_score']:.4f}"


def main():
    a = parse(); seed_all(a.run_seed)
    out_dir = os.path.join(B, a.out_name, a.mode + ("_smoke" if a.smoke else "")); os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "train_log.txt"), "a", encoding="utf-8")
    def log(s): print(s); logf.write(s + "\n"); logf.flush()
    dev = v3.DEVICE
    log(f"== {time.strftime('%Y-%m-%d %H:%M:%S')}  lora_train_v4  {vars(a)}")

    # ---- data: identical seeded split to v3 (same held-out images)
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

    # round-robin over ALL held-out images (same rule as the patched v3)
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

    cfg = dict(vars(a), train_strats=TRAIN_STRATS, unseen_strats=["EXT", "CPC"], selection_strats=SEL_STRATS,
               selection_rule="argmax over epochs of the mean over BB/CP/CPN/MP5 of macro-over-strata mIoU "
                              "(SAM's own mask choice, native resolution) on the fixed held-out TRAIN instances; "
                              "ties -> earlier epoch",
               train_images=len(tr_ids), heldout_images=len(val_ids), val_instances=n_v,
               written=time.strftime('%Y-%m-%d %H:%M:%S'))
    cfg_p = os.path.join(out_dir, "config.json")
    if not os.path.exists(cfg_p): json.dump(cfg, open(cfg_p, "w"), indent=2)

    # ---- model: LoRA on the image encoder + trainable confidence head
    sam = sam_model_registry["vit_h"](checkpoint=v3.SAM_CKPT).to(dev); sam.eval()
    if a.train in ("lora+head", "lora"):
        n_tr = inject_lora(sam, a.rank, a.alpha, iou_head=(a.train == "lora+head")); sam.to(dev)
    else:                                   # baselines without LoRA
        for p_ in sam.parameters(): p_.requires_grad_(False)
        mod = sam.mask_decoder.iou_prediction_head if a.train == "head" else sam.mask_decoder
        for p_ in mod.parameters(): p_.requires_grad_(True)
        n_tr = sum(p_.numel() for p_ in sam.parameters() if p_.requires_grad)
    enc_trainable = any(p_.requires_grad for p_ in sam.image_encoder.parameters())
    log(f"train variant '{a.train}', run seed {a.run_seed}: trainable parameters {n_tr:,} "
        f"(image encoder trainable: {enc_trainable})")
    tf = ResizeLongestSide(sam.image_encoder.img_size)

    # ---- SELF-TEST: training preprocessing must equal SamPredictor exactly
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
        r0 = evaluate(sam, val_items, a.mode); hist.append(dict(epoch=0, **r0))
        json.dump(hist, open(hist_p, "w"), indent=2); log(f"  epoch 0: {brief(r0)}")

    trng = random.Random(a.run_seed + 100)
    for ep in range(start_ep + 1, a.epochs + 1):
        t0, steps, loss_sum, mloss_sum, iloss_sum = time.time(), 0, 0.0, 0.0, 0.0
        used = Counter()
        order = tr_ids[:]; random.Random(a.run_seed + ep).shuffle(order)
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
                strat = trng.choice(TRAIN_STRATS)
                members, plist = window_prompts(members, strat, H, W, wx, wy)
                if not members: continue
                with torch.set_grad_enabled(enc_trainable):
                    emb, (ih, iw) = encode(sam, tf, np.ascontiguousarray(img[wy:wy + wh, wx:wx + ww]), dev)
                s = ih / wh
                pts = bt = None
                if plist[0]["points"] is not None:
                    c = tf.apply_coords(np.stack([p["points"] for p in plist]).astype(float), (wh, ww))
                    pts = (torch.as_tensor(c, dtype=torch.float, device=dev),
                           torch.as_tensor(np.stack([p["labels"] for p in plist]), dtype=torch.int, device=dev))
                if plist[0]["box"] is not None:
                    bt = torch.as_tensor(tf.apply_boxes(np.stack([p["box"] for p in plist]).astype(float), (wh, ww)),
                                         dtype=torch.float, device=dev)
                with torch.no_grad():
                    sparse, dense = sam.prompt_encoder(points=pts, boxes=bt, masks=None)
                low, iou_pred = sam.mask_decoder(image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
                                                 sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                                 multimask_output=True)
                logits = sam.postprocess_masks(low, (ih, iw), (ih, iw))          # (N, 3, ih, iw)
                gt = torch.as_tensor(np.stack([gt_at_input(x["segmentation"], wx, wy, s, ih, iw) for x in members]),
                                     device=dev)[:, None].expand_as(logits)
                per = F.binary_cross_entropy_with_logits(logits, gt, reduction="none").mean((-1, -2)) + dice_loss(logits, gt)
                mask_loss = per.min(1).values.mean()                              # best of 3 candidates
                with torch.no_grad():                                             # true IoU of every candidate
                    pm = (logits > 0).float()
                    inter = (pm * gt).sum((-1, -2)); union = pm.sum((-1, -2)) + gt.sum((-1, -2)) - inter
                    true_iou = inter / union.clamp(min=1.0)
                iou_loss = F.mse_loss(iou_pred, true_iou)
                loss = mask_loss + a.iou_w * iou_loss
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                steps += 1; loss_sum += loss.item(); mloss_sum += mask_loss.item(); iloss_sum += iou_loss.item()
                used[strat] += 1
                if steps % 100 == 0 or a.smoke:
                    log(f"  ep {ep} step {steps}/{a.windows_per_epoch}  loss {loss_sum/steps:.4f} "
                        f"(mask {mloss_sum/steps:.4f}, conf {iloss_sum/steps:.4f})  [{strat}]  "
                        f"lr {sched.get_last_lr()[0]:.2e}  {(time.time()-t0)/steps:.2f}s/step")
            del img
        ck = dict(lora=lora_state(sam), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, config=cfg)
        torch.save(ck, os.path.join(out_dir, f"epoch_{ep:03d}.pt"))
        r = evaluate(sam, val_items, a.mode)
        hist.append(dict(epoch=ep, train_loss=loss_sum / max(1, steps), prompt_counts=dict(used), **r))
        json.dump(hist, open(hist_p, "w"), indent=2)
        log(f"epoch {ep}: loss {loss_sum/max(1,steps):.4f}  prompts used {dict(used)}  ({(time.time()-t0)/60:.1f} min)")
        log(f"  val {brief(r)}")
        torch.cuda.empty_cache()

    best = max([h for h in hist if h["epoch"] > 0], key=lambda h: (h["selection_score"], -h["epoch"]))
    shutil.copy(os.path.join(out_dir, f"epoch_{best['epoch']:03d}.pt"), os.path.join(out_dir, "best.pt"))
    log(f"== selection rule -> epoch {best['epoch']} (selection score {best['selection_score']:.4f}; "
        f"epoch 0 baseline {hist[0]['selection_score']:.4f}) -> best.pt")


if __name__ == "__main__":
    main()
