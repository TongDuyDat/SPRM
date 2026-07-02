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
from torch.utils.data import Subset

from tools.utils import load_config, set_seed, setup_logger
from datasets import build_prepared_zsl_splits, build_zsl_splits
from model.hsprn_model import HSPRN
from losses import HSPRNLoss
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train HSPRN for ZSL benchmark")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to dataset-specific YAML config")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data root directory")
    parser.add_argument("--prepared_root", type=str, default=None,
                        help="Use a self-contained prepared_datasets root")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--semantic_embedding_path", type=str, default=None,
                        help="Path to cached class semantic embeddings (.pt/.npy/.npz/.mat)")
    parser.add_argument("--semantic_metadata_path", type=str, default=None,
                        help="JSON/.pt path containing class semantics with name_c/attr_c/hier_c/desc_c")
    parser.add_argument("--semantic_backbone", type=str, default=None,
                        help="Override semantic backbone: clip, tinyclip, sbert, bert")
    parser.add_argument("--semantic_pretrained_weights", type=str, default=None,
                        help="Override semantic text-only model path or Hugging Face id")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Train on the first N samples for pipeline smoke tests")
    parser.add_argument("--eval_subset_size", type=int, default=None,
                        help="Evaluate on the first N samples of each eval split")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override DataLoader worker count")
    return parser.parse_args()


def maybe_subset(dataset, size):
    if size is None:
        return dataset
    size = min(int(size), len(dataset))
    return Subset(dataset, range(size))


def resolve_semantic_source(cfg):
    data_cfg = cfg.data
    source = getattr(data_cfg, "semantic_source", "auto")
    if source is None:
        source = "auto"
    source = str(source).lower()
    if source == "auto":
        if getattr(data_cfg, "semantic_embedding_path", None):
            return "embedding"
        if getattr(data_cfg, "semantic_metadata_path", None):
            return "class_desc"
        return "attribute"
    allowed = {"attribute", "embedding", "class_desc"}
    if source not in allowed:
        raise ValueError(
            f"Unknown data.semantic_source={source!r}. Expected one of {sorted(allowed)}."
        )
    return source


def configure_semantic_input(cfg, attribute, semantic_source):
    semantic_cfg = cfg.model.semantic
    if semantic_source == "class_desc":
        backbone_type = getattr(semantic_cfg, "backbone_type", "embedding")
        if backbone_type in {"attribute", "identity", "embedding"}:
            raise ValueError(
                "semantic_source='class_desc' requires a text semantic backbone "
                "such as 'clip' or 'bert', not backbone_type='embedding'."
            )
        if getattr(semantic_cfg, "input_dim", None) is not None:
            raise ValueError(
                "semantic_source='class_desc' must keep model.semantic.input_dim null. "
                "Raw name_c/attr_c/hier_c/desc_c records are encoded by the text backbone."
            )
        shared_dim = getattr(semantic_cfg, "shared_dim", None)
        out_dim = getattr(cfg.model.aggregator, "out_dim", None)
        if shared_dim != out_dim:
            raise ValueError(
                "Full HSPRN loss requires model.semantic.shared_dim == "
                "model.aggregator.out_dim so Eq. (34) can compare v_tilde and "
                f"p_sem_y. Got shared_dim={shared_dim}, out_dim={out_dim}."
            )
        return

    cfg.model.semantic.input_dim = int(attribute.shape[1])
    if getattr(cfg.model.loss, "use_full", True):
        shared_dim = getattr(cfg.model.semantic, "shared_dim", None)
        if shared_dim != cfg.model.semantic.input_dim:
            print(
                "WARNING: Eq. (33) semantic consistency requires "
                f"shared_dim ({shared_dim}) == semantic input dim ({cfg.model.semantic.input_dim}). "
                "The loss will raise unless config is adjusted."
            )


def main():
    args = parse_args()

    # Load config (default + dataset-specific)
    default_cfg_path = os.path.join(os.path.dirname(args.config), "default.yaml")
    cfg = load_config(args.config, default_path=default_cfg_path)

    # CLI overrides
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.prepared_root:
        cfg.data.source = "prepared"
        cfg.data.prepared_root = args.prepared_root
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.semantic_embedding_path is not None:
        cfg.data.semantic_embedding_path = args.semantic_embedding_path
    if args.semantic_metadata_path is not None:
        cfg.data.semantic_metadata_path = args.semantic_metadata_path
    if args.semantic_backbone is not None:
        cfg.model.semantic.backbone_type = args.semantic_backbone
    if args.semantic_pretrained_weights is not None:
        cfg.model.semantic.pretrained_weights = args.semantic_pretrained_weights
    if args.subset_size is not None:
        cfg.data.subset_size = args.subset_size
    if args.eval_subset_size is not None:
        cfg.data.eval_subset_size = args.eval_subset_size
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

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
    data_source = str(getattr(cfg.data, "source", "legacy")).lower()
    prepared_root = getattr(cfg.data, "prepared_root", "./prepared_datasets")
    if data_source not in {"legacy", "prepared"}:
        raise ValueError("data.source must be 'legacy' or 'prepared'.")
    source_root = prepared_root if data_source == "prepared" else data_root
    print(f"Loading dataset: {dataset_name} from {source_root} ({data_source})")

    semantic_embedding_path = getattr(cfg.data, "semantic_embedding_path", None)
    semantic_metadata_path = getattr(cfg.data, "semantic_metadata_path", None)
    if data_source == "prepared" and getattr(cfg.data, "semantic_source", "auto") == "auto":
        cfg.data.semantic_source = "class_desc"
    semantic_source = resolve_semantic_source(cfg)
    cfg.data.semantic_source = semantic_source
    data_mode = getattr(cfg.data, "mode", "image")
    subset_size = getattr(cfg.data, "subset_size", None)
    eval_subset_size = getattr(cfg.data, "eval_subset_size", None)
    if (subset_size is not None or eval_subset_size is not None) and data_mode != "image":
        raise ValueError(
            "Subset smoke training must use raw images (`data.mode: image`), "
            f"not cached image features. Got data.mode={data_mode!r}."
        )
    image_preprocess = getattr(cfg.data, "image_preprocess", None)
    if image_preprocess is None:
        image_preprocess = "clip" if "clip" in cfg.model.vision.backbone_type.lower() else "imagenet"
    if data_source == "prepared":
        if data_mode != "image":
            raise ValueError("Prepared dataset packages currently support data.mode='image' only.")
        semantic_embedding_path = None
        semantic_metadata_path = os.path.join(
            prepared_root,
            dataset_name,
            "metadata",
            "class_semantics.json",
        )
        splits = build_prepared_zsl_splits(
            prepared_root,
            dataset_name,
            require_full_semantics=True,
            image_size=getattr(cfg.data, "image_size", 224),
            image_preprocess=image_preprocess,
        )
    else:
        splits = build_zsl_splits(
            data_root,
            dataset_name,
            semantic_embedding_path=semantic_embedding_path,
            semantic_metadata_path=semantic_metadata_path,
            require_full_semantics=True,
            mode=data_mode,
            image_root=getattr(cfg.data, "image_root", None),
            image_size=getattr(cfg.data, "image_size", 224),
            image_preprocess=image_preprocess,
        )
    trainval = splits["trainval"]
    test_seen = splits["test_seen"]
    test_unseen = splits["test_unseen"]
    attribute = splits["attribute"]
    seen_classes = splits["seen_classes"]
    unseen_classes = splits["unseen_classes"]

    trainval = maybe_subset(trainval, subset_size)
    test_seen = maybe_subset(test_seen, eval_subset_size)
    test_unseen = maybe_subset(test_unseen, eval_subset_size)

    print(f"  Data mode: {data_mode} (raw images -> visual backbone)")
    print(f"  Train: {len(trainval)} samples")
    print(f"  Test seen: {len(test_seen)} samples")
    print(f"  Test unseen: {len(test_unseen)} samples")
    print(f"  Seen classes: {len(seen_classes)}, Unseen: {len(unseen_classes)}")
    if semantic_source == "class_desc":
        print(f"  Semantic source: raw class-desc JSON ({semantic_metadata_path})")
    elif semantic_source == "embedding":
        print(f"  Semantic dim: {attribute.shape[1]} ({semantic_embedding_path})")
    else:
        print(f"  Semantic dim: {attribute.shape[1]} (xlsa17 attributes)")

    configure_semantic_input(cfg, attribute, semantic_source)

    # Model
    model = HSPRN(cfg)
    criterion = HSPRNLoss(cfg)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

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
