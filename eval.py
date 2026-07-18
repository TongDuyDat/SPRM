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
from tools.semantic_policy import (
    assert_full_after_llm_semantics,
    configure_semantic_input,
    resolve_semantic_source,
    unify_after_llm_enabled,
)
from datasets import build_prepared_zsl_splits, build_zsl_splits
from model.hsprn_model import HSPRN
from tools.evaluator import ZSLEvaluator
from torch.utils.data import DataLoader, Subset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HSPRN for ZSL benchmark")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--prepared_root", type=str, default=None,
                        help="Use a self-contained prepared_datasets root")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--semantic_embedding_path", type=str, default=None,
                        help="Path to cached class semantic embeddings (.pt/.npy/.npz/.mat)")
    parser.add_argument("--semantic_metadata_path", type=str, default=None,
                        help="JSON/.pt path containing class semantics with name_c/attr_c/hier_c/desc_c")
    parser.add_argument("--semantic_backbone", type=str, default=None,
                        help="Override semantic backbone: clip, tinyclip, sbert, bert")
    parser.add_argument("--semantic_pretrained_weights", type=str, default=None,
                        help="Override semantic text-only model path or Hugging Face id")
    parser.add_argument("--eval_subset_size", type=int, default=None,
                        help="Evaluate the first N samples from each test split")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override DataLoader worker count")
    return parser.parse_args()


def maybe_subset(dataset, size):
    if size is None:
        return dataset
    return Subset(dataset, range(min(int(size), len(dataset))))


def semantic_input(attribute, class_semantics, class_indices, device):
    if class_semantics is not None:
        return {
            str(int(class_index)): class_semantics[int(class_index)]
            for class_index in class_indices
        }
    return {
        str(int(class_index)): attribute[int(class_index)].to(device)
        for class_index in class_indices
    }


@torch.no_grad()
def compute_all_scores(
    model,
    loader,
    attribute,
    class_semantics,
    class_indices,
    num_total_classes,
    device,
):
    """Compute compatibility scores for all samples against given classes."""
    model.eval()
    all_scores = []
    all_labels = []

    attr = semantic_input(attribute, class_semantics, class_indices, device)

    for batch in loader:
        features, labels = batch[:2]
        features = features.to(device)
        out = model(attr, features)
        scores_subset = out.scores if hasattr(out, 'scores') else out

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
    if args.prepared_root:
        cfg.data.source = "prepared"
        cfg.data.prepared_root = args.prepared_root
    if args.semantic_embedding_path is not None:
        cfg.data.semantic_embedding_path = args.semantic_embedding_path
    if args.semantic_metadata_path is not None:
        cfg.data.semantic_metadata_path = args.semantic_metadata_path
    if args.semantic_backbone is not None:
        cfg.model.semantic.backbone_type = args.semantic_backbone
    if args.semantic_pretrained_weights is not None:
        cfg.model.semantic.pretrained_weights = args.semantic_pretrained_weights
    if args.eval_subset_size is not None:
        cfg.data.eval_subset_size = args.eval_subset_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    set_seed(cfg.training.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
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
    image_preprocess = getattr(cfg.data, "image_preprocess", None)
    if image_preprocess is None:
        image_preprocess = "clip" if "clip" in cfg.model.vision.backbone_type.lower() else "imagenet"
    if data_source == "prepared":
        if data_mode != "image":
            raise ValueError("Prepared dataset packages currently support data.mode='image' only.")
        semantic_embedding_path = None
        splits = build_prepared_zsl_splits(
            prepared_root,
            dataset_name,
            require_full_semantics=True,
            image_size=getattr(cfg.data, "image_size", 224),
            image_preprocess=image_preprocess,
        )
        semantic_metadata_path = str(
            splits["test_seen"].package_dir / "metadata" / "class_semantics.json"
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
    test_seen = splits["test_seen"]
    test_unseen = splits["test_unseen"]
    attribute = splits["attribute"]
    seen_classes = splits["seen_classes"]
    unseen_classes = splits["unseen_classes"]
    configure_semantic_input(cfg, attribute, semantic_source)
    class_semantics = None
    if semantic_source == "class_desc":
        class_semantics = test_seen.get_class_semantics()
        if unify_after_llm_enabled(cfg):
            assert_full_after_llm_semantics(
                class_semantics,
                num_total_classes=int(attribute.shape[0]),
                context="eval class_semantics",
            )

    eval_subset_size = getattr(cfg.data, "eval_subset_size", None)
    test_seen = maybe_subset(test_seen, eval_subset_size)
    test_unseen = maybe_subset(test_unseen, eval_subset_size)

    print(f"  Data mode: {data_mode} (raw images -> visual backbone)")
    print(f"  Test seen: {len(test_seen)} samples")
    print(f"  Test unseen: {len(test_unseen)} samples")
    print(f"  Seen classes: {len(seen_classes)}, Unseen: {len(unseen_classes)}")
    if semantic_source == "class_desc":
        print(f"  Semantic source: raw class-desc JSON ({semantic_metadata_path})")
    elif semantic_source == "embedding":
        print(f"  Semantic dim: {attribute.shape[1]} ({semantic_embedding_path})")
    else:
        print(f"  Semantic dim: {attribute.shape[1]} (class attributes)")
    if unify_after_llm_enabled(cfg):
        print("  Plan 3 unify_after_llm: ON (force class_desc, same encode as train)")

    bs = getattr(cfg.data, "batch_size", 64) * 2
    nw = getattr(cfg.data, "num_workers", 4)
    test_seen_loader = DataLoader(
        test_seen, batch_size=bs, shuffle=False, num_workers=nw,
        pin_memory=device == "cuda",
    )
    test_unseen_loader = DataLoader(
        test_unseen, batch_size=bs, shuffle=False, num_workers=nw,
        pin_memory=device == "cuda",
    )

    # Model
    model = HSPRN(cfg).to(device)
    ckpt = load_checkpoint(args.ckpt, model)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Evaluate
    evaluator = ZSLEvaluator(seen_classes, unseen_classes)
    num_total = attribute.shape[0]

    # Score all classes once; ZSL evaluation masks seen classes internally.
    all_cls = list(range(num_total))
    scores_seen_gzsl, labels_seen_gzsl = compute_all_scores(
        model, test_seen_loader, attribute, class_semantics,
        all_cls, num_total, device
    )
    scores_unseen_gzsl, labels_unseen_gzsl = compute_all_scores(
        model, test_unseen_loader, attribute, class_semantics,
        all_cls, num_total, device
    )
    zsl_T1 = evaluator.eval_zsl(scores_unseen_gzsl, labels_unseen_gzsl) * 100
    u, s, H = evaluator.eval_gzsl(
        scores_seen_gzsl, labels_seen_gzsl,
        scores_unseen_gzsl, labels_unseen_gzsl
    )
    cal_u, cal_s, cal_H, cal_gamma = evaluator.eval_gzsl_calibrated(
        scores_seen_gzsl, labels_seen_gzsl,
        scores_unseen_gzsl, labels_unseen_gzsl,
    )

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*50}")
    print(f"ZSL  T1:  {zsl_T1:.1f}%")
    print(f"GZSL u:   {u*100:.1f}%")
    print(f"GZSL s:   {s*100:.1f}%")
    print(f"GZSL H:   {H*100:.1f}%")
    print(f"Cal u:    {cal_u*100:.1f}%")
    print(f"Cal s:    {cal_s*100:.1f}%")
    print(f"Cal H:    {cal_H*100:.1f}% (gamma={cal_gamma:.2f})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
