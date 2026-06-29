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
    parser.add_argument("--semantic_embedding_path", type=str, default=None,
                        help="Path to cached class semantic embeddings (.pt/.npy/.npz/.mat)")
    parser.add_argument("--semantic_metadata_path", type=str, default=None,
                        help="JSON/.pt path containing class semantics with name_c/attr_c/hier_c/desc_c")
    return parser.parse_args()


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
                "Full HSPRN evaluation requires model.semantic.shared_dim == "
                "model.aggregator.out_dim. Got "
                f"shared_dim={shared_dim}, out_dim={out_dim}."
            )
        return

    cfg.model.semantic.input_dim = int(attribute.shape[1])


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
    if args.semantic_embedding_path is not None:
        cfg.data.semantic_embedding_path = args.semantic_embedding_path
    if args.semantic_metadata_path is not None:
        cfg.data.semantic_metadata_path = args.semantic_metadata_path

    set_seed(cfg.training.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Dataset
    semantic_embedding_path = getattr(cfg.data, "semantic_embedding_path", None)
    semantic_metadata_path = getattr(cfg.data, "semantic_metadata_path", None)
    semantic_source = resolve_semantic_source(cfg)
    cfg.data.semantic_source = semantic_source
    data_mode = getattr(cfg.data, "mode", "image")
    image_preprocess = getattr(cfg.data, "image_preprocess", None)
    if image_preprocess is None:
        image_preprocess = "clip" if "clip" in cfg.model.vision.backbone_type.lower() else "imagenet"
    splits = build_zsl_splits(
        cfg.data.data_root,
        cfg.data.dataset,
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
    all_classes = list(range(attribute.shape[0]))
    configure_semantic_input(cfg, attribute, semantic_source)
    class_semantics = None
    if semantic_source == "class_desc":
        class_semantics = test_seen.get_class_semantics()

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
        model, test_unseen_loader, attribute, class_semantics,
        unseen_classes, num_total, device
    )
    zsl_T1 = evaluator.eval_zsl(scores_unseen, labels_unseen) * 100

    # GZSL
    all_cls = list(range(num_total))
    scores_seen_gzsl, labels_seen_gzsl = compute_all_scores(
        model, test_seen_loader, attribute, class_semantics,
        all_cls, num_total, device
    )
    scores_unseen_gzsl, labels_unseen_gzsl = compute_all_scores(
        model, test_unseen_loader, attribute, class_semantics,
        all_cls, num_total, device
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
