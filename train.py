#!/usr/bin/env python
"""
HSPRN Training Entry Point.

Usage:
    python train.py --config config/awa2.yaml [--data_root /path/to/data]

The config file is merged with config/default.yaml as base.
Command-line --data_root overrides data.data_root in the config.
"""

import argparse
import os
import sys
import torch

from tools.utils import load_config, set_seed, setup_logger
from datasets import build_zsl_splits
from model.hsprn_model import HSPRN
from losses import HSPRNLoss
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train HSPRN for ZSL benchmark")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to dataset-specific YAML config")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data root directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--semantic_embedding_path", type=str, default=None,
                        help="Path to cached class semantic embeddings (.pt/.npy/.npz/.mat)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config (default + dataset-specific)
    default_cfg_path = os.path.join(os.path.dirname(args.config), "default.yaml")
    cfg = load_config(args.config, default_path=default_cfg_path)

    # CLI overrides
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.semantic_embedding_path is not None:
        cfg.data.semantic_embedding_path = args.semantic_embedding_path

    # Seed
    set_seed(cfg.training.seed)

    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Dataset
    dataset_name = cfg.data.dataset
    data_root = cfg.data.data_root
    print(f"Loading dataset: {dataset_name} from {data_root}")

    semantic_embedding_path = getattr(cfg.data, "semantic_embedding_path", None)
    splits = build_zsl_splits(
        data_root, dataset_name, semantic_embedding_path=semantic_embedding_path
    )
    trainval = splits["trainval"]
    test_seen = splits["test_seen"]
    test_unseen = splits["test_unseen"]
    attribute = splits["attribute"]
    seen_classes = splits["seen_classes"]
    unseen_classes = splits["unseen_classes"]

    print(f"  Train: {len(trainval)} samples")
    print(f"  Test seen: {len(test_seen)} samples")
    print(f"  Test unseen: {len(test_unseen)} samples")
    print(f"  Seen classes: {len(seen_classes)}, Unseen: {len(unseen_classes)}")
    semantic_source = semantic_embedding_path or "xlsa17 attributes"
    print(f"  Semantic dim: {attribute.shape[1]} ({semantic_source})")

    # Train-time semantic input is always a cached tensor. BERT/CLIP extraction
    # should happen offline with tools/extract_semantic_embeddings.py.
    cfg.model.semantic.input_dim = int(attribute.shape[1])

    # Model
    model = HSPRN(cfg)
    criterion = HSPRNLoss(cfg)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Trainer
    trainer = Trainer(
        cfg=cfg,
        model=model,
        criterion=criterion,
        train_dataset=trainval,
        test_seen_dataset=test_seen,
        test_unseen_dataset=test_unseen,
        attribute=attribute,
        seen_classes=seen_classes,
        unseen_classes=unseen_classes,
        device=device,
    )

    # Train
    best_H, best_epoch = trainer.train()
    print(f"\nFinal: Best H = {best_H:.1f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()
