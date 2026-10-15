"""Dataset builders behind ``data.type``.

The config names the builder:

    data: {type: "MultilabelSplit", dataset: "nuswide", ...}
    data: {type: "CZSLSplit",       dataset: "mit-states", ...}
    data: {type: "PreparedZSL",     dataset: "CUB", ...}

Every builder returns the same dict shape, which lets one engine drive all families:

    train, test_seen, test_unseen : torch Dataset
    attribute                     : Tensor [C, D] class semantics (phi)
    seen_classes, unseen_classes  : np.ndarray of class indices
    eval_classes                  : int, size of the scoring space
    label_format                  : "index" | "multihot"
    bench                         : CZSL only, the benchmark dict
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from tools.registry import DATASETS


def _phi_from_checkpoint(path):
    """Load a precomputed phi tensor, unwrapping ``{"embeddings": ...}`` / ``{"phi": ...}``."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "embeddings" in obj:
            obj = obj["embeddings"]
        elif "phi" in obj:
            obj = obj["phi"]
    return obj.float()


# ---------------------------------------------------------------------------
# Multi-label (MS-COCO, NUS-WIDE)
# ---------------------------------------------------------------------------

@DATASETS.register("MultilabelSplit")
def build_multilabel_split(cfg):
    """COCO / NUS-WIDE multi-label ZSL+GZSL splits.

    Chosen by ``cfg.data.dataset``. Returns multi-hot labels and an ``eval_classes``
    matching the full scoring space (65 for COCO, 1006 for NUS) so GZSL cannot
    silently degenerate to ZSL.
    """
    from datasets import ml_dataset

    name = str(cfg.data.dataset).lower()
    data_root = str(Path(cfg.data.data_root).expanduser())
    image_size = int(getattr(cfg.data, "image_size", 224))

    if name == "coco":
        test_bench = ml_dataset.build_coco_bench(data_root, image_size=image_size)
        train_bench = ml_dataset.build_coco_train(data_root, image_size=image_size)
        seen_idx = test_bench["seen_cls_idx"]
        unseen_idx = test_bench["unseen_cls_idx"]
        eval_idx = test_bench["eval_cls_idx"]

        train_ds, test_ds = train_bench["dataset_train"], test_bench["dataset_test"]

        n_seen, n_unseen = len(seen_idx), len(unseen_idx)
        eval_classes = len(eval_idx)
        # COCO targets are already [N, 65]; the 65 columns ARE eval_cls_idx.
        assert eval_classes == n_seen + n_unseen, (eval_classes, n_seen, n_unseen)
        # seen_idx/unseen_idx are raw COCO ids (0..79), but phi, the test targets and
        # the score columns live in eval POSITIONS (0..64). Convert once here so every
        # consumer indexes the same space. Train targets are [N, 48] in seen_idx order,
        # so seen positions must keep that order.
        pos = {c: j for j, c in enumerate(eval_idx)}
        seen_idx = [pos[c] for c in seen_idx]
        unseen_idx = [pos[c] for c in unseen_idx]
    elif name == "nuswide":
        bench = ml_dataset.build_nuswide_bench(data_root, image_size=image_size)
        test_ds = bench["dataset_test"]
        root = Path(data_root)
        train_ds = ml_dataset.build_nuswide_train(str(root), image_size)["dataset_train"]
        _, c81 = _nus_tag_lists(str(root / "Groundtruth"))
        n_seen, n_unseen = 925, len(c81)
        # 925 seen (TagList1k \ Concepts81) + 81 unseen = the 1006-class space.
        eval_classes = n_seen + n_unseen
        seen_idx = list(range(n_seen))
        unseen_idx = list(range(n_seen, eval_classes))
    else:
        raise ValueError(
            f"MultilabelSplit got dataset={name!r}; expected 'coco' or 'nuswide'."
        )

    # Class semantics over the FULL eval space, in eval-class order.
    if name == "coco":
        attribute = _phi_from_checkpoint(cfg.data.semantic_embedding_path)
        if attribute.shape[0] != eval_classes:
            raise ValueError(
                f"COCO phi {tuple(attribute.shape)} != {eval_classes} eval "
                f"classes ({cfg.data.semantic_embedding_path})."
            )
    else:
        # NUS needs two files: 925 seen tags + 81 unseen concepts. Concatenating in this
        # order is what keeps the columns aligned with the ground truth.
        phi_seen_path = cfg.data.seen_semantic_embedding_path
        phi_seen = _phi_from_checkpoint(phi_seen_path)
        phi_unseen = _phi_from_checkpoint(cfg.data.semantic_embedding_path)
        attribute = torch.cat([phi_seen, phi_unseen], dim=0)
        if attribute.shape[0] != 1006:
            raise ValueError(
                f"NUS phi must be 1006 classes (925 seen + 81 unseen), got "
                f"{tuple(attribute.shape)} from {phi_seen_path} + "
                f"{cfg.data.semantic_embedding_path}."
            )

    # The trainer needs the full eval-space ground truth, but the loaders only carry
    # their own label space (COCO: all 65; NUS: the 81 unseen). Attach the aligned
    # table to the datasets so it can be read back in loader row order.
    if name == "coco":
        eval_targets = _ml_targets(test_ds)
    else:
        eval_targets = ml_dataset.nus_gt_1006(test_ds, data_root)
    test_ds.eval_targets = eval_targets
    if train_ds is not test_ds and hasattr(train_ds, "__dict__"):
        train_ds.eval_targets = eval_targets

    return {
        "train": train_ds,
        "test_seen": test_ds,
        "test_unseen": test_ds,
        "attribute": attribute,
        "seen_classes": np.asarray(seen_idx, dtype=np.int64),
        "unseen_classes": np.asarray(unseen_idx, dtype=np.int64),
        "eval_classes": int(eval_classes),
        "label_format": "multihot",
    }


def _ml_targets(ds):
    """Multi-hot targets for every row of ``ds``, in row order (COCO only)."""
    return np.stack([np.asarray(ds[i][1], dtype=np.float32)
                     for i in range(len(ds))])


def _nus_tag_lists(anno_dir):
    """925 seen tags (TagList1k \\ Concepts81) and the 81 unseen concepts."""
    tag1k = [l.strip() for l in open(os.path.join(anno_dir, "TagList1k.txt")) if l.strip()]
    c81 = [l.strip() for l in open(os.path.join(anno_dir, "Concepts81.txt")) if l.strip()]
    c81set = set(c81)
    seen925 = [t for t in tag1k if t not in c81set]
    assert len(seen925) == 925, len(seen925)
    assert len(c81) == 81, len(c81)
    return seen925, c81


# ---------------------------------------------------------------------------
# CZSL (MIT-States, UT-Zappos, C-GQA)
# ---------------------------------------------------------------------------

@DATASETS.register("CZSLSplit")
def build_czsl_split(cfg):
    """Compositional ZSL splits; one (state, object) pair per class."""
    from datasets.czsl_dataset import build_czsl_bench

    name = str(cfg.data.dataset)
    data_root = str(Path(cfg.data.data_root).expanduser())

    bench = build_czsl_bench(name, data_root, universe="closed")
    train_ds = bench["dataset_train"]
    seen_ds = bench["dataset_test_seen"]
    unseen_ds = bench["dataset_test_unseen"]

    attribute = _phi_from_checkpoint(cfg.data.semantic_embedding_path)
    if attribute.shape[0] != bench["num_pairs"]:
        raise ValueError(
            f"phi {tuple(attribute.shape)} != num_pairs {bench['num_pairs']} for "
            f"{name}. Regenerate semantics/phi for the full pair set."
        )

    return {
        "train": train_ds,
        "test_seen": seen_ds,
        "test_unseen": unseen_ds,
        "attribute": attribute,
        "seen_classes": np.asarray(bench["seen_classes"], dtype=np.int64),
        "unseen_classes": np.asarray(bench["unseen_classes"], dtype=np.int64),
        "eval_classes": int(bench["num_pairs"]),
        "label_format": "index",
        "bench": bench,
    }


# ---------------------------------------------------------------------------
# Classic ZSL / GZSL (CUB, AWA2, APY, SUN)
# ---------------------------------------------------------------------------

@DATASETS.register("PreparedZSL")
def build_prepared_zsl(cfg):
    """Classic ZSL splits from a prepared dataset package."""
    from datasets.prepared_dataset import build_prepared_zsl_splits

    if str(getattr(cfg.data, "source", "")).lower() != "prepared":
        raise ValueError(
            f"PreparedZSL needs data.source == 'prepared', got {getattr(cfg.data, 'source', None)!r}."
        )
    splits = build_prepared_zsl_splits(
        prepared_root=cfg.data.prepared_root,
        dataset_name=cfg.data.dataset,
        require_full_semantics=True,
        semantic_embedding_path=cfg.data.semantic_embedding_path,
        image_size=int(getattr(cfg.data, "image_size", 224)),
    )
    attribute = splits["attribute"]
    # The semantic encoder is the parameter-free embedding backbone over precomputed phi.
    cfg.model.semantic.input_dim = int(attribute.shape[1])

    return {
        "train": splits["trainval"],
        "test_seen": splits["test_seen"],
        "test_unseen": splits["test_unseen"],
        "attribute": attribute,
        "seen_classes": np.asarray(splits["seen_classes"], dtype=np.int64),
        "unseen_classes": np.asarray(splits["unseen_classes"], dtype=np.int64),
        "eval_classes": int(attribute.shape[0]),
        "label_format": "index",
    }
