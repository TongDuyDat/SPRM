#!/usr/bin/env python3
"""
HSPRN-A score fusion (R4): combine the compatibility scores of two trained
models at inference time.

Usage:
    python tools/fusion_eval.py \
        --config_a config/cub_clip_v5_r2_full.yaml \
        --ckpt_a checkpoints/cub_clip_v5_r2_full/CUB/best.pth \
        --config_b config/cub_clip_v5_r0_probe.yaml \
        [--beta_step 0.05] [--device cuda]

Model A is loaded from a checkpoint; model B is evaluated at initialization
(the R0 probe config has everything frozen, so no checkpoint is needed).
Fused score: s = beta * s_A + (1 - beta) * s_B, then calibrated stacking.
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import Subset

from tools.local_pkg import import_local_datasets
from tools.utils import load_config
from tools.semantic_policy import configure_semantic_input, resolve_semantic_source

_ds = import_local_datasets()
build_prepared_zsl_splits = _ds.build_prepared_zsl_splits

from model.hsprn_model import HSPRN
from losses import HSPRNLoss
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="HSPRN-A score fusion")
    parser.add_argument("--config_a", type=str, required=True)
    parser.add_argument("--ckpt_a", type=str, required=True)
    parser.add_argument("--config_b", type=str, required=True,
                        help="Frozen-geometry config (no checkpoint needed)")
    parser.add_argument("--beta_step", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_trainer(config_path, ckpt_path=None, device="cuda"):
    """Mirror train.py's setup up to Trainer construction (no training)."""
    default_cfg_path = os.path.join(os.path.dirname(config_path), "default.yaml")
    cfg = load_config(config_path, default_path=default_cfg_path)

    cfg.data.semantic_source = resolve_semantic_source(cfg)
    image_preprocess = getattr(cfg.data, "image_preprocess", None)
    if image_preprocess is None:
        image_preprocess = (
            "clip" if "clip" in cfg.model.vision.backbone_type.lower()
            else "imagenet"
        )
    splits = build_prepared_zsl_splits(
        cfg.data.prepared_root,
        cfg.data.dataset,
        require_full_semantics=True,
        image_size=getattr(cfg.data, "image_size", 224),
        image_preprocess=image_preprocess,
    )
    configure_semantic_input(cfg, splits["attribute"], cfg.data.semantic_source)

    model = HSPRN(cfg)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        # strict=False: CLIP backbones may save auxiliary buffers
        # (e.g. positional_embedding_new from pos-embed interpolation).
        result = model.load_state_dict(state["model_state_dict"], strict=False)
        if result.missing_keys:
            raise RuntimeError(
                f"Checkpoint is missing required keys: {result.missing_keys[:10]}"
            )
        if result.unexpected_keys:
            print(f"(ignoring {len(result.unexpected_keys)} auxiliary "
                  f"checkpoint keys, e.g. {result.unexpected_keys[0]})")
        print(f"Loaded checkpoint: {ckpt_path} "
              f"(epoch {state.get('epoch')}, best_H {state.get('best_H', 0):.1f})")

    criterion = HSPRNLoss(cfg)
    trainer = Trainer(
        cfg=cfg,
        model=model,
        criterion=criterion,
        train_dataset=splits["trainval"],
        test_seen_dataset=splits["test_seen"],
        test_unseen_dataset=splits["test_unseen"],
        attribute=splits["attribute"],
        seen_classes=splits["seen_classes"],
        unseen_classes=splits["unseen_classes"],
        device=device,
    )
    return trainer


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=== Building model A (trained) ===")
    trainer_a = build_trainer(args.config_a, args.ckpt_a, device)
    print("=== Building model B (frozen geometry) ===")
    trainer_b = build_trainer(args.config_b, None, device)

    all_classes = trainer_a.evaluator.all_classes
    evaluator = trainer_a.evaluator

    scores_seen_a, labels_seen = trainer_a._compute_scores(
        trainer_a.test_seen_loader, all_classes, "A eval seen"
    )
    scores_unseen_a, labels_unseen = trainer_a._compute_scores(
        trainer_a.test_unseen_loader, all_classes, "A eval unseen"
    )
    scores_seen_b, labels_seen_b = trainer_b._compute_scores(
        trainer_b.test_seen_loader, all_classes, "B eval seen"
    )
    scores_unseen_b, labels_unseen_b = trainer_b._compute_scores(
        trainer_b.test_unseen_loader, all_classes, "B eval unseen"
    )
    if not torch.equal(labels_seen, labels_seen_b) or \
            not torch.equal(labels_unseen, labels_unseen_b):
        raise RuntimeError("Label mismatch between the two models' loaders.")

    print(f"\n{'beta':>6} | {'u':>6} {'s':>6} {'H':>6} {'gamma':>6} | "
          f"{'ZSL':>6}")
    print("-" * 55)
    best = {"H": -1.0}
    beta = 0.0
    while beta <= 1.0 + 1e-9:
        fused_seen = beta * scores_seen_a + (1.0 - beta) * scores_seen_b
        fused_unseen = beta * scores_unseen_a + (1.0 - beta) * scores_unseen_b

        cal_u, cal_s, cal_H, cal_gamma = evaluator.eval_gzsl_calibrated(
            fused_seen, labels_seen, fused_unseen, labels_unseen
        )
        zsl = evaluator.eval_zsl(fused_unseen, labels_unseen)
        print(f"{beta:>6.2f} | {cal_u*100:>6.1f} {cal_s*100:>6.1f} "
              f"{cal_H*100:>6.1f} {cal_gamma:>6.2f} | {zsl*100:>6.1f}")
        if cal_H * 100 > best["H"]:
            best = {
                "H": cal_H * 100, "u": cal_u * 100, "s": cal_s * 100,
                "gamma": cal_gamma, "beta": beta, "ZSL": zsl * 100,
            }
        beta += args.beta_step

    print("-" * 55)
    print(f"BEST fusion: beta={best['beta']:.2f} -> Cal H={best['H']:.1f}% "
          f"(u={best['u']:.1f}, s={best['s']:.1f}, gamma={best['gamma']:.2f}, "
          f"ZSL={best['ZSL']:.1f}%)")


if __name__ == "__main__":
    main()
