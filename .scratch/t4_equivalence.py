"""Task 4 — numerical equivalence: CLIPVisualBackbone (visual-only) vs full CLIP.

Verifies:
  1. forward output identical to CLIP.encode_image.
  2. extract_levels identical to a manual run of the visual tower.
  3. parameter count drops by exactly the CLIP text-branch size.
  4. no complex dtype; autograd flows through the visual tower.
Runs on CPU with the local ViT-B/32 checkpoint (same code path as B/16).
"""
import warnings

warnings.filterwarnings("ignore")
import torch

from model.sematics.clip import load

PATH = "pre-trained/ViT-B-32.pt"
LEVELS = [4, 8, 12]
torch.manual_seed(0)
x = torch.randn(2, 3, 224, 224)


def unwrap(out):
    """Mirror CLIPVisualBackbone.forward's tuple handling exactly."""
    if isinstance(out, tuple):
        if len(out) >= 2 and out[1].dim() == 2:
            out = out[1]
        else:
            out = out[0]
            if out.dim() == 3:
                out = out[:, 0]
    return out.float()


def manual_extract_levels(visual, x, levels, dtype):
    wanted = {int(l) for l in levels}
    x = x.type(dtype)
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    ce = visual.class_embedding.to(dtype)
    x = torch.cat(
        [ce + torch.zeros(x.shape[0], 1, ce.shape[-1], dtype=dtype, device=x.device), x],
        dim=1,
    )
    x = x + visual.positional_embedding.to(dtype)
    x = visual.ln_pre(x)
    outs = []
    for i, blk in enumerate(visual.transformer.resblocks, start=1):
        x = blk(x)
        if i in wanted:
            outs.append(x.float())
    return outs


# ---------- Reference: full CLIP ----------
ref, _ = load(PATH, device="cpu", jit=False)
ref_dtype = ref.dtype
full_params = sum(p.numel() for p in ref.parameters())
visual_params = sum(p.numel() for p in ref.visual.parameters())
text_params = full_params - visual_params
ref_conv1 = ref.visual.conv1.weight.detach().clone()

with torch.no_grad():
    ref_out = unwrap(ref.encode_image(x))
    ref_levels = manual_extract_levels(ref.visual, x, LEVELS, ref_dtype)
del ref

# ---------- New visual-only backbone ----------
from model.vision import CLIPVisualBackbone

bb = CLIPVisualBackbone(model_name=PATH, pretrained=True)
bb_params = sum(p.numel() for p in bb.parameters())

with torch.no_grad():
    bb_out = bb(x)
    bb_levels = bb.extract_levels(x, LEVELS)

# ---------- Checks ----------
ok = True

# weights identical (same checkpoint loaded)
same_w = torch.equal(ref_conv1, bb.visual.conv1.weight.detach())
print(f"[{'OK' if same_w else 'FAIL'}] visual weights identical to full-CLIP load")
ok &= same_w

# 1. forward equivalence
fwd = torch.allclose(ref_out, bb_out, atol=0.0, rtol=0.0)
max_fwd = (ref_out - bb_out).abs().max().item()
print(f"[{'OK' if fwd else 'FAIL'}] forward == encode_image  (max|diff|={max_fwd:.3e})")
ok &= fwd

# 2. extract_levels equivalence
for lv, a, b in zip(LEVELS, ref_levels, bb_levels):
    eq = a.shape == b.shape and torch.allclose(a, b, atol=0.0, rtol=0.0)
    md = (a - b).abs().max().item() if a.shape == b.shape else float("nan")
    print(f"[{'OK' if eq else 'FAIL'}] extract_levels block {lv}: shape={tuple(b.shape)} max|diff|={md:.3e}")
    ok &= eq

# 3. parameter drop == text branch
drop = full_params - bb_params
match = bb_params == visual_params
print(f"[{'OK' if match else 'FAIL'}] backbone params == visual-only params")
print(f"      full CLIP   : {full_params:,}")
print(f"      visual tower: {visual_params:,}")
print(f"      text branch : {text_params:,}")
print(f"      backbone now: {bb_params:,}  (dropped {drop:,})")
ok &= match and (drop == text_params)

# 4. no complex dtype + autograd flows
complex_any = any(p.is_complex() for p in bb.parameters())
print(f"[{'FAIL' if complex_any else 'OK'}] no complex dtype in backbone params")
ok &= not complex_any
print(f"      backbone dtype property = {bb.dtype}")

bb.train()
out = bb(x)
out.sum().backward()
grad_ok = bb.visual.conv1.weight.grad is not None and torch.isfinite(bb.visual.conv1.weight.grad).all()
print(f"[{'OK' if grad_ok else 'FAIL'}] autograd flows to visual tower (conv1 grad finite)")
ok &= grad_ok

print("\nRESULT:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
