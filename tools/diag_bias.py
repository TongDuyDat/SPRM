#!/usr/bin/env python
"""
Diagnostic: where does the residual seen-bias live? (E3/E5/E7 checkpoints)

Loads a trained checkpoint and reports, for both test sets:
  - mean cosine score to GT / other seen / all unseen prototypes
  - global offset: mean score to seen prototypes vs unseen prototypes
  - per-class mean score (hub analysis: which classes attract most mass)
  - W_s deviation from identity (isometry vs rotation diagnosis)

Usage:
    python tools/diag_bias.py --config config/...yaml --ckpt checkpoints/.../best.pth
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from tools.local_pkg import import_local_datasets
from tools.utils import load_config, set_seed, load_checkpoint
from tools.semantic_policy import (
    assert_full_after_llm_semantics,
    configure_semantic_input,
    resolve_semantic_source,
    unify_after_llm_enabled,
)

_ds = import_local_datasets()
build_prepared_zsl_splits = _ds.build_prepared_zsl_splits
build_zsl_splits = _ds.build_zsl_splits

from model.hsprn_model import HSPRN
from torch.utils.data import DataLoader, Subset
from eval import compute_all_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap samples per test set (speed).")
    return parser.parse_args()


def group_stats(scores, labels, seen_set, unseen_set):
    """Return mean score split by GT / other-seen / unseen per sample."""
    gt = scores.gather(1, labels.unsqueeze(1)).squeeze(1)

    seen_list = sorted(seen_set)
    unseen_list = sorted(unseen_set)
    seen_cols = scores[:, seen_list]
    unseen_cols = scores[:, unseen_list]

    # Global GT label -> column index in the seen-only view (seen samples only).
    seen_pos = {c: i for i, c in enumerate(seen_list)}
    gt_in_seen = torch.tensor(
        [seen_pos.get(int(l), -1) for l in labels.cpu()], device=scores.device
    )
    seen_other = seen_cols.clone()
    rows = (gt_in_seen >= 0).nonzero(as_tuple=True)[0]
    if rows.numel() > 0:
        seen_other[rows, gt_in_seen[rows]] = -float("inf")

    return {
        "gt": gt.mean().item(),
        "other_seen": seen_other.max(dim=1)[0].mean().item(),
        "mean_seen": seen_cols.mean().item(),
        "max_unseen": unseen_cols.max(dim=1)[0].mean().item(),
        "mean_unseen": unseen_cols.mean().item(),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config, default_path=str(
        Path(args.config).parent / "default.yaml"))
    cfg.data.source = "prepared"
    set_seed(cfg.training.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    prepared_root = getattr(cfg.data, "prepared_root", "./prepared_datasets")
    semantic_source = resolve_semantic_source(cfg)
    cfg.data.semantic_source = semantic_source
    image_preprocess = getattr(cfg.data, "image_preprocess", None)
    if image_preprocess is None:
        image_preprocess = "clip" if "clip" in cfg.model.vision.backbone_type.lower() else "imagenet"

    splits = build_prepared_zsl_splits(
        prepared_root, cfg.data.dataset, require_full_semantics=True,
        image_size=getattr(cfg.data, "image_size", 224),
        image_preprocess=image_preprocess,
    )
    semantic_metadata_path = str(
        splits["test_seen"].package_dir / "metadata" / "class_semantics.json"
    )
    test_seen = splits["test_seen"]
    test_unseen = splits["test_unseen"]
    attribute = splits["attribute"]
    seen_classes = splits["seen_classes"]
    unseen_classes = splits["unseen_classes"]
    configure_semantic_input(cfg, attribute, semantic_source)
    class_semantics = test_seen.get_class_semantics()
    if unify_after_llm_enabled(cfg):
        assert_full_after_llm_semantics(
            class_semantics, num_total_classes=int(attribute.shape[0]),
            context="diag_bias class_semantics",
        )

    if args.max_samples:
        n = min(args.max_samples, len(test_seen))
        test_seen = Subset(test_seen, range(n))
        n = min(args.max_samples, len(test_unseen))
        test_unseen = Subset(test_unseen, range(n))

    bs = getattr(cfg.data, "batch_size", 64) * 2
    test_seen_loader = DataLoader(test_seen, batch_size=bs, shuffle=False, num_workers=4)
    test_unseen_loader = DataLoader(test_unseen, batch_size=bs, shuffle=False, num_workers=4)

    model = HSPRN(cfg).to(device)
    load_checkpoint(args.ckpt, model)
    model.eval()

    num_total = attribute.shape[0]
    all_cls = list(range(num_total))
    phi_cache = model.psc.encode_class_phi(class_semantics).to(device)

    scores_seen, labels_seen = compute_all_scores(
        model, test_seen_loader, attribute, class_semantics,
        all_cls, num_total, device, phi_cache=phi_cache,
    )
    scores_unseen, labels_unseen = compute_all_scores(
        model, test_unseen_loader, attribute, class_semantics,
        all_cls, num_total, device, phi_cache=phi_cache,
    )

    seen_set = set(int(c) for c in seen_classes)
    unseen_set = set(int(c) for c in unseen_classes)

    print(f"\n{'='*64}\nDIAG  {args.ckpt}\n{'='*64}")

    for name, sc, lb in (("test_seen", scores_seen, labels_seen),
                         ("test_unseen", scores_unseen, labels_unseen)):
        st = group_stats(sc, lb, seen_set, unseen_set)
        print(f"\n[{name}]")
        print(f"  score to GT:            {st['gt']:+.4f}")
        print(f"  max score other seen:   {st['other_seen']:+.4f}")
        print(f"  max score unseen:       {st['max_unseen']:+.4f}")
        print(f"  mean score seen group:  {st['mean_seen']:+.4f}")
        print(f"  mean score unseen grp:  {st['mean_unseen']:+.4f}")
        print(f"  GLOBAL OFFSET (seen-unseen mean): "
              f"{st['mean_seen'] - st['mean_unseen']:+.4f}")

    per_class = scores_unseen.mean(dim=0)
    seen_mass = per_class[list(seen_set)].mean().item()
    unseen_mass = per_class[list(unseen_set)].mean().item()
    print(f"\n[hub analysis on test_unseen sample scores]")
    print(f"  mean unseen-sample score over seen classes:   {seen_mass:+.4f}")
    print(f"  mean unseen-sample score over unseen classes: {unseen_mass:+.4f}")
    top = torch.topk(per_class, 5)
    print("  top-5 hub classes (unseen samples):")
    for idx, val in zip(top.indices.tolist(), top.values.tolist()):
        tag = "SEEN" if idx in seen_set else "unseen"
        print(f"    class {idx} ({tag}): {val:+.4f}")

    ws = model.psc.Ws.weight.detach()
    diff = ws - torch.eye(ws.shape[0], device=ws.device)
    print(f"\n[Ws diagnosis]")
    print(f"  ||Ws - I||_F:  {diff.norm().item():.4f}")
    print(f"  ||Ws||_F:      {ws.norm().item():.4f}")
    sigma = torch.linalg.svdvals(ws)
    print(f"  svd(Ws): min {sigma.min().item():.4f} / median "
          f"{sigma.median().item():.4f} / max {sigma.max().item():.4f}")

    text_input = {str(int(c)): phi_cache[int(c)] for c in all_cls}
    p = model.psc(text_input)["p_sem_norm"].detach()
    gram = p @ p.T
    seen_rows = gram[list(seen_set)][:, list(seen_set)]
    off = seen_rows - torch.eye(len(seen_set), device=gram.device)
    print(f"  mean |cos| among seen prototypes (off-diag): "
          f"{off.abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
