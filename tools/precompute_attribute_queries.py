#!/usr/bin/env python
"""Precompute the frozen attribute query bank for the v10 cross-attention IPR.

Encodes every attribute name of a dataset through the CLIP text tower (the
exact ``psc._encode_texts`` path used by precompute_class_phi / class_desc
training) and saves ``{"embeddings": [n_attr, D]}`` to the .pt file referenced
by ``model.refiner.attribute_query_path``.

The v10 training configs use ``semantic.backbone_type: "embedding"`` (the text
tower is dropped at runtime), so this tool builds ONLY a PriSematicConstructor
whose semantic section mirrors the class_desc configs (backbone_type "clip",
input_dim null) — the same encoder + tokenizer the class phi files were made
with. Building the full HSPRN here would be circular anyway: the v10 refiner
requires this very .pt file to exist at construction time.

Attribute-name text is lightly cleaned for CLIP ("::" and "_" → spaces) since
published identifiers (e.g. "has_bill_shape::dagger") are not natural language.
The raw names are stored alongside for traceability.

Usage (on IBM, needs the prepared dataset):
    python tools/precompute_attribute_queries.py \
        --config config/cub_clip_v10_ca_b16.yaml \
        --output pre-trained/cub_attrq_b16.pt
"""

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F

from tools.utils import load_config

from model.sematics.constructor import PriSematicConstructor


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute the attribute query bank for CrossAttentionPrototypeRefiner")
    p.add_argument("--config", type=str, required=True,
                   help="YAML config of the dataset (used to resolve dataset name, "
                        "prepared_root and the CLIP checkpoint path)")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--prepared_root", type=str, default=None)
    p.add_argument("--names", type=str, default=None,
                   help="Override the attribute names JSON "
                        "(default: prepared_root/<dataset>/metadata/attribute_names.json)")
    p.add_argument("--output", type=str, required=True, help="Output .pt path")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for the (frozen) text encode; cpu is fine")
    return p.parse_args()


def clean_attribute_text(name):
    """Published attribute identifier -> short natural-language phrase."""
    text = str(name).strip()
    text = text.replace("::", " ")
    # "has_bill_shape" -> "has bill shape"; keep raw word order.
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def build_text_encoder_cfg(cfg):
    """Semantic-only cfg mirroring the class_desc configs (CLIP text tower).

    input_dim must be None so PriSematicConstructor keeps backbone_type
    "clip" (a set input_dim forces the identity-embedding path). shared_dim
    matches the CLIP ViT-B/16 text output (512).
    """
    import types

    clip_model = getattr(cfg.model.vision, "clip_model", "./pre-trained/ViT-B-16.pt")
    semantic = types.SimpleNamespace(
        freeze=True,
        backbone_type="clip",
        pretrained=True,
        clip_model=clip_model,
        max_text_len=77,
        input_dim=None,
        shared_dim=512,
    )
    model = types.SimpleNamespace(semantic=semantic)
    return types.SimpleNamespace(model=model)


def main():
    args = parse_args()

    default_cfg_path = os.path.join(os.path.dirname(args.config), "default.yaml")
    cfg = load_config(args.config, default_path=default_cfg_path)
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.prepared_root:
        cfg.data.source = "prepared"
        cfg.data.prepared_root = args.prepared_root

    dataset_name = cfg.data.dataset
    prepared_root = getattr(cfg.data, "prepared_root", "./prepared_datasets")
    names_path = args.names or os.path.join(
        prepared_root, dataset_name, "metadata", "attribute_names.json")
    with open(names_path, "r", encoding="utf-8") as handle:
        raw_names = [str(v) for v in json.load(handle)]
    if not raw_names:
        raise ValueError(f"Empty attribute name list: {names_path}")
    print(f"{dataset_name}: {len(raw_names)} attribute names from {names_path}")

    texts = [clean_attribute_text(n) for n in raw_names]

    # Only the frozen CLIP text tower is needed; the semantic-only cfg above
    # guarantees the same encoder/tokenizer as the class_desc phi precompute.
    psc = PriSematicConstructor(build_text_encoder_cfg(cfg)).to(args.device)
    psc.eval()
    emb = psc._encode_texts(texts).detach().float().cpu()
    if emb.dim() != 2 or emb.shape[0] != len(texts):
        raise RuntimeError(
            f"Text encode returned {tuple(emb.shape)}, expected "
            f"({len(texts)}, D).")
    emb = F.normalize(emb, p=2, dim=-1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": emb,
        "names": raw_names,
        "texts": texts,
        "dataset": dataset_name,
        "source": "PriSematicConstructor._encode_texts (CLIP text tower)",
    }, out_path)
    print(f"Saved attribute query bank {tuple(emb.shape)} -> {out_path}")
    print(f"  norm mean={emb.norm(dim=-1).mean().item():.4f} (unit-normalized)")
    print(f"  example: {raw_names[0]!r} -> {texts[0]!r}")


if __name__ == "__main__":
    main()
