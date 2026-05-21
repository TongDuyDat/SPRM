#!/usr/bin/env python
"""
HSPRN Evaluation Entry Point.

Usage:
    python eval.py --config config/awa2.yaml --ckpt checkpoints/AWA2/best.pth
"""

import argparse
import os
import torch

from tools.utils import load_config, set_seed, load_checkpoint
from datasets import build_zsl_splits
from model.hsprn_model import HSPRN
from tools.evaluator import ZSLEvaluator
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HSPRN for ZSL benchmark")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


@torch.no_grad()
def compute_all_scores(model, loader, attribute, class_indices, num_total_classes, device):
    """Compute compatibility scores for all samples against given classes."""
    model.eval()
    all_scores = []
    all_labels = []

    attr = attribute[class_indices].to(device)

    for features, labels in loader:
        features = features.to(device)
        scores_subset = model(attr, features)

        scores_full = torch.full(
            (features.size(0), num_total_classes),
            -float("inf"), device=device
        )
        for i, c in enumerate(class_indices):
            scores_full[:, c] = scores_subset[:, i]

        all_scores.append(scores_full.cpu())
        all_labels.append(labels)

    return torch.cat(all_scores, dim=0), torch.cat(all_labels, dim=0)


def main():
    args = parse_args()

    default_cfg_path = os.path.join(os.path.dirname(args.config), "default.yaml")
    cfg = load_config(args.config, default_path=default_cfg_path)

    if args.data_root:
        cfg.data.data_root = args.data_root

    set_seed(cfg.training.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Dataset
    splits = build_zsl_splits(cfg.data.data_root, cfg.data.dataset)
    test_seen = splits["test_seen"]
    test_unseen = splits["test_unseen"]
    attribute = splits["attribute"]
    seen_classes = splits["seen_classes"]
    unseen_classes = splits["unseen_classes"]
    all_classes = list(range(attribute.shape[0]))

    bs = getattr(cfg.data, "batch_size", 64) * 2
    test_seen_loader = DataLoader(test_seen, batch_size=bs, shuffle=False, num_workers=4)
    test_unseen_loader = DataLoader(test_unseen, batch_size=bs, shuffle=False, num_workers=4)

    # Model
    model = HSPRN(cfg).to(device)
    ckpt = load_checkpoint(args.ckpt, model)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Evaluate
    evaluator = ZSLEvaluator(seen_classes, unseen_classes)
    num_total = attribute.shape[0]

    # ZSL
    scores_unseen, labels_unseen = compute_all_scores(
        model, test_unseen_loader, attribute, unseen_classes, num_total, device
    )
    zsl_T1 = evaluator.eval_zsl(scores_unseen, labels_unseen) * 100

    # GZSL
    all_cls = list(range(num_total))
    scores_seen_gzsl, labels_seen_gzsl = compute_all_scores(
        model, test_seen_loader, attribute, all_cls, num_total, device
    )
    scores_unseen_gzsl, labels_unseen_gzsl = compute_all_scores(
        model, test_unseen_loader, attribute, all_cls, num_total, device
    )
    u, s, H = evaluator.eval_gzsl(
        scores_seen_gzsl, labels_seen_gzsl,
        scores_unseen_gzsl, labels_unseen_gzsl
    )

    print(f"\n{'='*50}")
    print(f"Dataset: {cfg.data.dataset}")
    print(f"{'='*50}")
    print(f"ZSL  T1:  {zsl_T1:.1f}%")
    print(f"GZSL u:   {u*100:.1f}%")
    print(f"GZSL s:   {s*100:.1f}%")
    print(f"GZSL H:   {H*100:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
