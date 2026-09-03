#!/usr/bin/env python
"""T9 — verify the precomputed-text-embedding path (drop CLIP text tower).

Checks (all on CPU with local ViT-B/32 weights; semantic stays 512-dim so the
mechanism is identical to ViT-B/16):
  1. Model builds from the *_precomputed.yaml config with a parameter-free
     semantic backbone (NO CLIP text tower).
  2. Total params drop to ~91.0M (B/32)  [~89.4M extrapolated for B/16],
     learnable unchanged at 1,081,856.
  3. Trainer sets _phi_cache from the precomputed attribute (embedding branch),
     so lambda_trans=0.5 does NOT hard-raise and _build_trans_caches completes.
  4. One full train_one_epoch runs end-to-end (forward + all phi-cache losses +
     backward + optimizer step) with finite loss and updated learnable params.
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np
import torch

from tools.utils import load_config
from model.hsprn_model import HSPRN
from losses import HSPRNLoss
from trainer import Trainer

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "cpu"
IMG = 224
D = 512
C_TOTAL = 10
SEEN = list(range(6))
UNSEEN = list(range(6, C_TOTAL))

results = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f"  -> {detail}" if detail else ""))
    return cond


# ---------------------------------------------------------------------------
# Config (precomputed variant) + local CPU overrides
# ---------------------------------------------------------------------------
cfg = load_config(
    "config/cub_clip_v8_hsprn_b16_precomputed.yaml",
    default_path="config/default.yaml",
)
cfg.model.vision.backbone_type = "clip_vit_b32"
cfg.model.vision.clip_model = "./pre-trained/ViT-B-32.pt"
cfg.data.batch_size = 4
cfg.data.num_workers = 0
cfg.data.drop_last = False

check("config semantic_source == embedding", cfg.data.semantic_source == "embedding",
      str(cfg.data.semantic_source))
check("config unify_after_llm == False",
      bool(getattr(cfg.experiment, "unify_after_llm", False)) is False)
check("config lambda_trans == 0.5 (user value)",
      float(cfg.model.loss.lambda_trans) == 0.5)
check("config semantic_embedding_path set",
      bool(getattr(cfg.data, "semantic_embedding_path", None)))

# ---------------------------------------------------------------------------
# Mock precomputed phi_t (attribute) — stands in for the .pt produced on IBM
# ---------------------------------------------------------------------------
attribute = torch.randn(C_TOTAL, D)

# ---------------------------------------------------------------------------
# 1+2. Model build + param accounting (no text tower)
# ---------------------------------------------------------------------------
model = HSPRN(cfg)

psc = model.psc
enc_params = sum(p.numel() for p in psc.encoder.parameters())
check("psc.backbone_type == embedding", getattr(psc, "backbone_type", None) == "embedding",
      str(getattr(psc, "backbone_type", None)))
check("psc.encoder is parameter-free (text tower dropped)", enc_params == 0,
      f"psc.encoder params={enc_params:,}")
check("psc has NO CLIP text model inside", not hasattr(psc.encoder, "model"),
      f"hasattr model={hasattr(psc.encoder, 'model')}")

total = sum(p.numel() for p in model.parameters())
learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
vis = sum(p.numel() for p in model.vis_encoder.parameters())
ws = sum(p.numel() for p in psc.Ws.parameters())
mfa = sum(p.numel() for p in model.mfa.parameters())
ipr = sum(p.numel() for p in model.ipr.parameters())

print("\n--- Param accounting (B/32 local build, embedding mode) ---")
print(f"  vis_encoder : {vis:>13,}")
print(f"  psc.encoder : {enc_params:>13,}  (identity, was 63,428,096 in class_desc)")
print(f"  psc.Ws      : {ws:>13,}")
print(f"  mfa         : {mfa:>13,}")
print(f"  ipr         : {ipr:>13,}")
print(f"  TOTAL       : {total:>13,}  (~{total/1e6:.1f}M)")
print(f"  LEARNABLE   : {learnable:>13,}")

# B/32 expectation: vis 87,849,216 + 0 + Ws 32,768 + mfa 2,097,152 + ipr 1,049,088
check("total params ~91.0M (B/32 embedding mode)", 89_000_000 < total < 93_000_000,
      f"{total:,}")
check("learnable unchanged == 1,081,856", learnable == 1_081_856, f"{learnable:,}")
# Sanity: dropping the text tower removed exactly the CLIP text params vs class_desc
check("text-tower removal saves ~63.4M vs class_desc (154.46M)",
      abs((154_456_320 - total) - 63_428_096) < 2_000_000,
      f"delta={154_456_320 - total:,}")

# B/16 extrapolation reported to user
b16_total = total - vis + 86_192_640
print(f"  [extrapolated B/16 total] ~{b16_total:,} (~{b16_total/1e6:.1f}M)")

# ---------------------------------------------------------------------------
# 3+4. Trainer construction (embedding _phi_cache branch) + one epoch
# ---------------------------------------------------------------------------
class TinyDS(torch.utils.data.Dataset):
    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.randn(3, IMG, IMG), int(self.labels[i])


train_ds = TinyDS([c for c in SEEN for _ in range(2)])
seen_ds = TinyDS([c for c in SEEN])
unseen_ds = TinyDS([c for c in UNSEEN])

criterion = HSPRNLoss(cfg)

try:
    trainer = Trainer(
        cfg=cfg, model=model, criterion=criterion,
        train_dataset=train_ds, test_seen_dataset=seen_ds,
        test_unseen_dataset=unseen_ds, attribute=attribute,
        seen_classes=np.array(SEEN), unseen_classes=np.array(UNSEEN),
        device=DEVICE,
    )
    trainer_ok = True
except Exception as e:  # noqa: BLE001
    trainer_ok = False
    check("Trainer constructed (lambda_trans=0.5, embedding mode)", False, repr(e))
    raise

check("Trainer constructed (lambda_trans=0.5 did NOT raise)", trainer_ok)
check("_phi_cache populated from precomputed attribute",
      trainer._phi_cache is not None and torch.allclose(trainer._phi_cache, attribute),
      f"shape={None if trainer._phi_cache is None else tuple(trainer._phi_cache.shape)}")
check("_build_trans_caches completed (_teacher set)",
      trainer._teacher is not None and trainer._level_means is not None,
      f"teacher={None if trainer._teacher is None else tuple(trainer._teacher.shape)}, "
      f"level_means={None if trainer._level_means is None else tuple(trainer._level_means.shape)}")

# Capture learnable params before the step to confirm the optimizer updates them.
ws_before = [p.detach().clone() for p in psc.Ws.parameters()]
ipr_before = [p.detach().clone() for p in model.ipr.parameters() if p.requires_grad]

stats = trainer.train_one_epoch(0)
loss_val = stats.get("loss", float("nan"))
check("train_one_epoch completed with finite loss", np.isfinite(loss_val),
      f"loss={loss_val:.4f}, components={ {k: round(v, 4) for k, v in stats.items() if k != 'loss'} }")

ws_changed = any(not torch.equal(a, b) for a, b in
                 zip(ws_before, [p.detach() for p in psc.Ws.parameters()]))
ipr_changed = any(not torch.equal(a, b) for a, b in
                  zip(ipr_before, [p.detach() for p in model.ipr.parameters() if p.requires_grad]))
check("Ws updated by optimizer", ws_changed)
check("IPR updated by optimizer", ipr_changed)

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
n_pass = sum(1 for r in results if r[0] == "PASS")
print(f"T9 RESULT: {n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    print("FAILED CHECKS:")
    for status, name, detail in results:
        if status == "FAIL":
            print(f"  - {name}: {detail}")
    sys.exit(1)
print("ALL PASS")
