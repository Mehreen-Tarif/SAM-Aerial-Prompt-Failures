"""
lora_sam.py -- LoRA for SAM ViT-H image-encoder attention (q and v projections).
Shared by lora_train_v3.py (training) and sam_eval_v4.py --lora (evaluation), so the
model trained is exactly the model evaluated.

rank 8, alpha 16 on q and v of all 32 blocks = 32 x 4 x 8 x 1280 = 1,310,720 parameters
(the same budget as the earlier LoRA runs).
"""
import torch
import torch.nn as nn

class LoRAQKV(nn.Module):
    """Wraps a block's fused qkv Linear; adds low-rank updates to the q and v slices only."""
    def __init__(self, qkv: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.qkv, self.d, self.r, self.scale = qkv, qkv.in_features, r, alpha / r
        assert qkv.out_features == 3 * self.d, "unexpected qkv shape"
        dev, dt = qkv.weight.device, qkv.weight.dtype
        self.A_q = nn.Parameter(torch.empty(r, self.d, device=dev, dtype=dt))
        self.A_v = nn.Parameter(torch.empty(r, self.d, device=dev, dtype=dt))
        self.B_q = nn.Parameter(torch.zeros(self.d, r, device=dev, dtype=dt))   # B = 0 -> starts as the base model
        self.B_v = nn.Parameter(torch.zeros(self.d, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A_q, a=5 ** 0.5); nn.init.kaiming_uniform_(self.A_v, a=5 ** 0.5)

    def forward(self, x):
        out, d = self.qkv(x), self.d
        dq = (x @ self.A_q.t()) @ self.B_q.t() * self.scale
        dv = (x @ self.A_v.t()) @ self.B_v.t() * self.scale
        return torch.cat([out[..., :d] + dq, out[..., d:2 * d], out[..., 2 * d:] + dv], dim=-1)

def inject_lora(sam, r=8, alpha=16, iou_head=False):
    """Freeze everything, then add LoRA to every image-encoder block. Returns #trainable params.
    iou_head=True also makes the mask decoder's IoU-prediction (confidence) head trainable (lora_train_v4)."""
    for p in sam.parameters():
        p.requires_grad_(False)
    for blk in sam.image_encoder.blocks:
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, r, alpha)
    if iou_head:
        for p in sam.mask_decoder.iou_prediction_head.parameters():
            p.requires_grad_(True)
    return sum(p.numel() for p in sam.parameters() if p.requires_grad)

def lora_state(sam):
    return {k: v.detach().cpu().clone() for k, v in sam.named_parameters() if v.requires_grad}

def load_lora(sam, path):
    """Load a fine-tuning checkpoint into a (LoRA-injected) SAM. Works for every variant:
    v3 LoRA, v4 LoRA + confidence head, head-only and decoder-only baselines. Fails loudly on any mismatch."""
    ck = torch.load(path, map_location="cpu")
    sd = ck["lora"] if "lora" in ck else ck
    params = dict(sam.named_parameters())
    unknown = set(sd) - set(params)
    if unknown:
        raise RuntimeError(f"checkpoint has {len(unknown)} tensors the model does not have, e.g. {sorted(unknown)[:3]}")
    lora_keys = {k for k in params if k.rsplit(".", 1)[-1] in ("A_q", "A_v", "B_q", "B_v")}
    if lora_keys & set(sd) and not lora_keys <= set(sd):
        raise RuntimeError(f"LoRA checkpoint mismatch: {len(lora_keys - set(sd))} LoRA tensors missing")
    with torch.no_grad():
        for k, v in sd.items():
            params[k].copy_(v.to(params[k].device, params[k].dtype))
            params[k].requires_grad_(True)          # marks what was fine-tuned (lora_state saves exactly these)
    return ck
