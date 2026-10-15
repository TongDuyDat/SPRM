#!/usr/bin/env python
"""
SPRN (LPR) training entry point.

Usage:
    python train.py --dataset cub
    python train.py --dataset coco --smoke

--dataset picks config/datasets/<d>.yaml; its family (classic | czsl | multilabel)
picks config/families/<f>.yaml; --method picks config/methods/<m>.yaml (default lpr).
Command-line flags override the composed config.
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import yaml
from torch.utils.data import Subset

from tools.registry import DATASETS, TRAINERS
from tools.utils import compose_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train SPRN (LPR) for ZSL benchmarks")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (config/datasets/<name>.yaml)")
    parser.add_argument("--method", type=str, default="lpr",
                        help="Method name (config/methods/<name>.yaml). Default: lpr")
    parser.add_argument("--data_root", type=str, default=None, help="Override data root directory")
    parser.add_argument("--prepared_root", type=str, default=None,
                        help="Override the prepared_datasets root (classic datasets)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Train on the first N samples for pipeline smoke tests")
    parser.add_argument("--eval_subset_size", type=int, default=None,
                        help="Evaluate on the first N samples of each eval split")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override DataLoader worker count")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override one config key, e.g. --set training.lr=5e-4 "
                             "(value parsed as YAML; repeatable)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Save to ./checkpoints/<method>/<run_name> and ./logs/<method>/<run_name> "
                             "(default: the dataset name)")
    parser.add_argument("--smoke", action="store_true",
                        help="Pipeline test: 3 epochs (1 per stage), ~8 steps/epoch, "
                             "writes to ./checkpoints/smoke/<dataset>. Scores are meaningless.")
    return parser.parse_args()


def maybe_subset(dataset, size):
    if size is None:
        return dataset
    size = min(int(size), len(dataset))
    return Subset(dataset, range(size))


def main():
    args = parse_args()

    cfg, _ = compose_config(args.dataset, args.method)
    print(f"  config: method={args.method} dataset={args.dataset}")

    # CLI overrides
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.prepared_root:
        cfg.data.prepared_root = args.prepared_root
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.subset_size is not None:
        cfg.data.subset_size = args.subset_size
    if args.eval_subset_size is not None:
        cfg.data.eval_subset_size = args.eval_subset_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    for item in args.set:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            if not hasattr(node, part):
                setattr(node, part, type(cfg)({}))
            node = getattr(node, part)
        setattr(node, leaf, yaml.safe_load(raw))
        print(f"  --set {key} = {getattr(node, leaf)!r}")
    if args.run_name:
        cfg.training.save_dir = f"./checkpoints/{args.method}/{args.run_name}"
        cfg.training.log_dir = f"./logs/{args.method}/{args.run_name}"

    if args.smoke:
        # One epoch per stage (stage 1: epoch 1, stage 2: epoch 2, stage 3: epoch 3).
        cfg.training.epochs = 3
        cfg.training.warmup_epochs = 1
        for stage, last in (("1", 1), ("2", 2)):
            getattr(cfg.training.phases, stage).epochs = last
        # ~8 optimizer steps per epoch: with 1 step, AMP's GradScaler can skip it
        # (fp16 overflow) and a whole stage trains nothing while looking fine.
        cfg.data.subset_size = 8 * int(cfg.data.batch_size)
        cfg.data.eval_subset_size = 512
        cfg.training.amp = False
        cfg.data.num_workers = min(int(cfg.data.num_workers), 4)
        cfg.training.save_dir = f"./checkpoints/smoke/{args.dataset}"
        cfg.training.log_dir = f"./logs/smoke/{args.dataset}"

    set_seed(cfg.training.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Dataset: chosen by data.type through the registry.
    import datasets.builders  # noqa: F401  (registers the builders)

    data_type = cfg.data.type
    built = DATASETS.build(data_type, cfg)

    trainval = built["train"]
    test_seen = built["test_seen"]
    test_unseen = built["test_unseen"]
    attribute = built["attribute"]
    seen_classes = built["seen_classes"]
    unseen_classes = built["unseen_classes"]

    trainval = maybe_subset(trainval, getattr(cfg.data, "subset_size", None))
    eval_subset_size = getattr(cfg.data, "eval_subset_size", None)
    test_seen = maybe_subset(test_seen, eval_subset_size)
    test_unseen = maybe_subset(test_unseen, eval_subset_size)

    print(f"  data.type: {data_type} (label_format={built['label_format']})")
    print(f"  Train: {len(trainval)} samples")
    print(f"  Test seen: {len(test_seen)} samples")
    print(f"  Test unseen: {len(test_unseen)} samples")
    print(f"  Seen classes: {len(seen_classes)}, Unseen: {len(unseen_classes)}")
    print(f"  Eval space: {built['eval_classes']} classes")
    print(f"  Semantic dim: {attribute.shape[1]} ({cfg.data.semantic_embedding_path})")

    # Model + criterion. (The multi-label trainer swaps the criterion for ASL itself.)
    from model.sprn_model import SPRN
    from losses import SPRNLoss

    model = SPRN(cfg)
    criterion = SPRNLoss(cfg)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Trainer: one class per benchmark family, chosen by trainer.type.
    import trainer  # noqa: F401  (registers the trainers)

    print(f"  trainer.type: {cfg.trainer.type}")
    trainer_cls = TRAINERS.get(cfg.trainer.type)
    run = trainer_cls(
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

    best, best_epoch = run.train()
    print(f"\nFinal: best = {best:.1f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()
