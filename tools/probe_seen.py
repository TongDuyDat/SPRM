#!/usr/bin/env python3
"""
Seen-class ceiling probe (HSPRN-A v7, Phase 1).

Extracts frozen CLIP image features for trainval + test_seen, then trains a
Linear or MLP probe to classify the 150 seen classes. Reports the achievable
seen accuracy of the backbone under a discriminative (non-prototype) head.

Usage:
    python tools/probe_seen.py --backbone clip_vit_b16 \
        --clip_model ./pre-trained/ViT-B-16.pt [--probe mlp] [--epochs 15]
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.local_pkg import import_local_datasets
from tools.utils import load_config, set_seed

_ds = import_local_datasets()
build_prepared_zsl_splits = _ds.build_prepared_zsl_splits

from model.vision_model import VisionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Seen-class ceiling probe")
    parser.add_argument("--backbone", type=str, default="clip_vit_b32",
                        choices=["clip_vit_b32", "clip_vit_b16", "clip_vit_l14"])
    parser.add_argument("--clip_model", type=str, default=None,
                        help="Path to the CLIP .pt weights")
    parser.add_argument("--prepared_root", type=str, default="./prepared_datasets")
    parser.add_argument("--dataset", type=str, default="CUB")
    parser.add_argument("--probe", type=str, default="mlp",
                        choices=["linear", "mlp"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cache_dir", type=str, default="./cache/probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_vision(backbone_type, clip_model):
    """Minimal vision wrapper reusing VisionModel's CLIP pooled path."""
    import types
    cfg = types.SimpleNamespace()
    cfg.feature_level = "pool"
    cfg.model = types.SimpleNamespace()
    cfg.model.vision = types.SimpleNamespace(
        backbone_type=backbone_type,
        pretrained=True,
        freeze=True,
        levels=None,
        clip_model=clip_model,
    )
    return VisionModel(cfg)


@torch.no_grad()
def extract_features(model, loader, device):
    feats, labels = [], []
    for batch in loader:
        images, y = batch[:2]
        images = images.to(device)
        out = model(images)
        if isinstance(out, (list, tuple)):
            out = model.mfa.gap(out[-1]) if out[-1].dim() > 2 else out[-1]
        feats.append(out.float().cpu())
        labels.append(y)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.clip_model is None:
        name = {"clip_vit_b32": "ViT-B-32", "clip_vit_b16": "ViT-B-16",
                "clip_vit_l14": "ViT-L-14"}[args.backbone]
        args.clip_model = f"./pre-trained/{name}.pt"

    # Dataset (prepared package, CLIP preprocessing)
    splits = build_prepared_zsl_splits(
        args.prepared_root, args.dataset,
        require_full_semantics=True, image_size=224,
        image_preprocess="clip",
    )
    seen_set = set(int(c) for c in splits["seen_classes"])
    label_map = {int(c): i for i, c in enumerate(sorted(seen_set))}
    num_seen = len(label_map)

    train_ds = splits["trainval"]
    test_ds = splits["test_seen"]
    # Deterministic extraction: use the eval transform for both splits.
    if hasattr(train_ds, "_default_transform"):
        train_ds.transform = train_ds._default_transform("test")
    if hasattr(test_ds, "_default_transform"):
        test_ds.transform = test_ds._default_transform("test")

    nw = 8
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=nw)

    # Feature cache per backbone
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, f"{args.dataset}_{args.backbone}.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached features: {cache_path}")
        data = torch.load(cache_path)
        train_feats, train_labels = data["train_feats"], data["train_labels"]
        test_feats, test_labels = data["test_feats"], data["test_labels"]
    else:
        print(f"Building vision model: {args.backbone}")
        model = build_vision(args.backbone, args.clip_model).to(device).eval()
        print("Extracting trainval features ...")
        train_feats, train_labels = extract_features(model, train_loader, device)
        print("Extracting test_seen features ...")
        test_feats, test_labels = extract_features(model, test_loader, device)
        torch.save({"train_feats": train_feats, "train_labels": train_labels,
                    "test_feats": test_feats, "test_labels": test_labels},
                   cache_path)
        print(f"Cached features -> {cache_path} "
              f"(dim={train_feats.shape[1]})")
        del model
        torch.cuda.empty_cache()

    # Map labels to 0..S-1 (seen only)
    train_y = torch.tensor([label_map[int(l)] for l in train_labels])
    test_y = torch.tensor([label_map[int(l)] for l in test_labels])

    # Standardize features (probe best practice)
    mu = train_feats.mean(dim=0, keepdim=True)
    sd = train_feats.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_feats = (train_feats - mu) / sd
    test_feats = (test_feats - mu) / sd

    dim = train_feats.shape[1]
    if args.probe == "linear":
        head = nn.Linear(dim, num_seen)
    else:
        head = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_seen),
        )
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    train_feats, train_y = train_feats.to(device), train_y.to(device)
    test_feats, test_y = test_feats.to(device), test_y.to(device)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(train_feats.shape[0], device=device)
        total = 0.0
        for i in range(0, train_feats.shape[0], args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = head(train_feats[idx])
            loss = F.cross_entropy(logits, train_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        sched.step()

        head.eval()
        with torch.no_grad():
            pred = head(test_feats).argmax(dim=1).cpu()
        acc = (pred == test_y.cpu()).float().mean().item() * 100
        best_acc = max(best_acc, acc)
        print(f"Epoch [{epoch}/{args.epochs}] loss={total/train_feats.shape[0]:.4f} "
              f"test_seen top-1={acc:.2f}% (best={best_acc:.2f}%)")

    print(f"\n=== PROBE RESULT [{args.backbone} / {args.probe}] ===")
    print(f"Seen-class ceiling: {best_acc:.2f}% top-1 on test_seen ({num_seen} classes)")


if __name__ == "__main__":
    main()
