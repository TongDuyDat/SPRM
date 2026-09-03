import warnings

warnings.filterwarnings("ignore")
import torch
from omegaconf import OmegaConf

from model.hsprn_model import HSPRN

# Load the v8 config but point at the local B/32 checkpoint (B/16 not downloaded).
cfg = OmegaConf.load("config/cub_clip_v8_hsprn_b16.yaml")
cfg.model.vision.backbone_type = "clip_vit_b32"
cfg.model.vision.clip_model = "./pre-trained/ViT-B-32.pt"
cfg.model.semantic.clip_model = "./pre-trained/ViT-B-32.pt"

torch.manual_seed(0)
model = HSPRN(cfg)
model.eval()


def cnt(m):
    return sum(p.numel() for p in m.parameters())


def cnt_train(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


print("=== per-submodule params (B/32 build) ===")
for name in ("psc", "vis_encoder", "mfa", "ipr", "rp_bank", "msh"):
    m = getattr(model, name)
    print(f"  {name:12s} total={cnt(m):>13,}  trainable={cnt_train(m):>11,}")

total = cnt(model)
trainable = cnt_train(model)
print(f"\n  FULL MODEL   total={total:,}  trainable={trainable:,}")

# Break down psc (text tower vs Ws) and vis_encoder (visual backbone).
print("\n=== breakdown ===")
print(f"  psc.encoder (CLIP text tower) : {cnt(model.psc.encoder):,}  (frozen)")
print(f"  psc.Ws                        : {cnt(model.psc.Ws):,}  (learnable)")
print(f"  vis_encoder (CLIP visual)     : {cnt(model.vis_encoder):,}  (frozen, visual-only after T4)")
print(f"  ipr                           : {cnt(model.ipr):,}  (learnable)")
print(f"  mfa                           : {cnt(model.mfa):,}")
print(f"  msh                           : {cnt(model.msh):,}")

# The <150M figure quoted in the plan excludes the frozen text tower.
visual_plus_learnable = cnt(model.vis_encoder) + cnt(model.psc.Ws) + cnt(model.ipr) + cnt(model.mfa) + cnt(model.msh)
print(f"\n  visual + learnable (excl. text tower) = {visual_plus_learnable:,}")
print(f"  text tower (frozen)                   = {cnt(model.psc.encoder):,}")

# requires_grad audit
print("\n=== requires_grad audit (learnable tensors) ===")
learnable_names = [n for n, p in model.named_parameters() if p.requires_grad]
groups = {}
for n in learnable_names:
    top = n.split(".")[0]
    groups[top] = groups.get(top, 0) + 1
for k, v in sorted(groups.items()):
    print(f"  {k:12s} {v} learnable tensors")
print("  learnable names sample:", learnable_names[:8])
