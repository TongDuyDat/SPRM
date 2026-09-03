#!/usr/bin/env python
"""Precompute class phi_t (pre-Ws CLIP text embeddings) for HSPRN.

Produces the file consumed by ``data.semantic_embedding_path`` so the frozen
CLIP text tower can be dropped from the runtime model (~152.8M -> ~89.4M, B/16).

Guarantee: the phi_t saved here is EXACTLY what ``Trainer`` caches via
``model.psc.encode_class_phi`` in class_desc mode, because this script drives
the very same code path — it mirrors train.py's dataset block, builds HSPRN from
the same class_desc config, and calls ``psc.encode_class_phi`` on the same
``get_class_semantics()`` records. Run it with the class_desc config, then train
with the companion *_precomputed.yaml config.

Usage (on IBM, needs the prepared dataset):
    python tools/precompute_class_phi.py \
        --config config/cub_clip_v8_hsprn_b16.yaml \
        --output pre-trained/cub_v8_phi_b16.pt
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from tools.local_pkg import import_local_datasets
from tools.utils import load_config
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


def parse_args():
    p = argparse.ArgumentParser(description="Precompute class phi_t for HSPRN")
    p.add_argument("--config", type=str, required=True,
                   help="class_desc YAML config (the same one used to train)")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--prepared_root", type=str, default=None)
    p.add_argument("--output", type=str, required=True, help="Output .pt path")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for the (frozen) text encode; cpu is fine")
    p.add_argument("--center", action="store_true",
                   help="B2: Phi <- normalize(Phi - mean_c Phi). Removes the near-constant "
                        "direction shared by every class (the component that carried no "
                        "discriminative signal yet dominated the cosine geometry).")
    p.add_argument("--field_norm", action="store_true",
                   help="B1: L2-normalize each field vector before mixing (model.semantic.field_norm).")
    p.add_argument("--field_weights", type=str, default=None,
                   help="B5: 4 comma-separated floats for name_c,attr_c,hier_c,desc_c "
                        "(model.semantic.field_weights). Only valid with --field_norm.")
    p.add_argument("--export_sentences", action="store_true",
                   help="v9.6 E2: additionally store per-sentence embeddings "
                        "embeddings_sentences [C,S,D] + sentence_mask [C,S] so eval "
                        "can score per-descriptor and average scores (CuPL-style).")
    p.add_argument("--control", choices=("none", "random_words", "name_repeat"),
                   default="none",
                   help="v9.6 E2 WaffleCLIP control: 'random_words' replaces every "
                        "attr/hier/desc sentence with class-agnostic random words from a "
                        "fixed wordlist (seed 42); 'name_repeat' keeps only the class name "
                        "repeated to match the sentence count. The class-level 'embeddings' "
                        "is recomputed the same way so bank + ensemble stay consistent.")
    p.add_argument("--metadata", type=str, default=None,
                   help="v9.6 E3: override the semantic records JSON (default: "
                        "prepared_root/<dataset>/metadata/class_semantics.json).")
    return p.parse_args()


WAFFLE_WORDLIST = [
    "quiet", "amber", "tufted", "ornate", "husky", "brisk", "sober", "vivid",
    "plush", "hollow", "angular", "meek", "raw", "nimble", "dappled", "coarse",
    "shy", "brash", "muted", "lean", "frosty", "unkempt", "sturdy", "lucid",
    "pallid", "briskly", "drab", "hale", "obscure", "tawny", "rugged", "vague",
    "prim", "stalwart", "flecked", "wry", "mealy", "austere", "broad", "deep",
    "curved", "plain", "sharp", "soft", "warm", "cold", "dull", "keen", "pale",
    "stark",
]


def build_control_sentences(sentences_per_class, control, rng):
    """Return (control_sentence_lists, control_records).

    random_words: every non-name sentence is replaced by ONE random pair of
    class-agnostic words from the fixed wordlist (WaffleCLIP-style); the name
    sentence is kept so the class identity stays (WaffleCLIP keeps the class
    name and appends random words around it).
    name_repeat: only the name sentence, repeated to the original sentence
    count (tests pure averaging/ensemble effect with zero extra semantics).
    """
    new_lists = []
    for sentences in sentences_per_class:
        if not sentences:
            raise ValueError("empty sentence list for a class")
        name = sentences[0]
        if control == "name_repeat":
            new_lists.append([name] * len(sentences))
        elif control == "random_words":
            words = [rng.choice(WAFFLE_WORDLIST) for _ in range(2)]
            filler = f"{name}, which has {' '.join(words)}."
            new_lists.append([name] + [filler] * (len(sentences) - 1))
        else:
            raise ValueError(f"unknown control {control!r}")
    return new_lists


def mean_offdiag_cos(matrix):
    gram = F.normalize(matrix.detach().float(), p=2, dim=-1)
    cos = gram @ gram.T
    off_diag = ~torch.eye(cos.shape[0], dtype=torch.bool)
    return float(cos[off_diag].mean())


def unwrap(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def main():
    args = parse_args()

    default_cfg_path = os.path.join(os.path.dirname(args.config), "default.yaml")
    cfg = load_config(args.config, default_path=default_cfg_path)
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.prepared_root:
        cfg.data.source = "prepared"
        cfg.data.prepared_root = args.prepared_root
    # B1/B5 act inside the encoding path, so they must be set here (phi is baked
    # once) rather than in the training config that later loads the .pt file.
    cfg.model.semantic.field_norm = bool(args.field_norm)
    if args.field_weights:
        weights = [float(x) for x in args.field_weights.split(",")]
        if len(weights) != 4:
            raise ValueError(
                "--field_weights expects 4 floats: name_c,attr_c,hier_c,desc_c")
        if not args.field_norm:
            raise ValueError("--field_weights requires --field_norm: with raw field "
                             "norms the weights do not control the mixture.")
        cfg.model.semantic.field_weights = weights

    # --- Dataset block: mirrors train.py exactly --------------------------
    dataset_name = cfg.data.dataset
    data_root = cfg.data.data_root
    data_source = str(getattr(cfg.data, "source", "legacy")).lower()
    prepared_root = getattr(cfg.data, "prepared_root", "./prepared_datasets")
    if data_source not in {"legacy", "prepared"}:
        raise ValueError("data.source must be 'legacy' or 'prepared'.")

    semantic_embedding_path = getattr(cfg.data, "semantic_embedding_path", None)
    semantic_metadata_path = getattr(cfg.data, "semantic_metadata_path", None)
    if data_source == "prepared" and getattr(cfg.data, "semantic_source", "auto") == "auto":
        cfg.data.semantic_source = "class_desc"
    semantic_source = resolve_semantic_source(cfg)
    cfg.data.semantic_source = semantic_source
    if semantic_source != "class_desc":
        raise ValueError(
            "precompute_class_phi must run in class_desc mode (it encodes the raw "
            f"name_c/attr_c/hier_c/desc_c records); got semantic_source={semantic_source!r}."
        )

    data_mode = getattr(cfg.data, "mode", "image")
    image_preprocess = getattr(cfg.data, "image_preprocess", None)
    if image_preprocess is None:
        image_preprocess = "clip" if "clip" in cfg.model.vision.backbone_type.lower() else "imagenet"

    if data_source == "prepared":
        semantic_embedding_path = None
        semantic_metadata_path = os.path.join(
            prepared_root, dataset_name, "metadata", "class_semantics.json")
        splits = build_prepared_zsl_splits(
            prepared_root, dataset_name,
            require_full_semantics=True,
            semantics_override_path=args.metadata or None,
            image_size=getattr(cfg.data, "image_size", 224),
            image_preprocess=image_preprocess,
        )
    else:
        splits = build_zsl_splits(
            data_root, dataset_name,
            semantic_embedding_path=semantic_embedding_path,
            semantic_metadata_path=semantic_metadata_path,
            require_full_semantics=True,
            mode=data_mode,
            image_root=getattr(cfg.data, "image_root", None),
            image_size=getattr(cfg.data, "image_size", 224),
            image_preprocess=image_preprocess,
        )
    trainval = splits["trainval"]
    attribute = splits["attribute"]

    configure_semantic_input(cfg, attribute, semantic_source)

    source_dataset = unwrap(trainval)
    if not hasattr(source_dataset, "get_class_semantics"):
        raise ValueError("Dataset has no get_class_semantics(); cannot precompute phi.")
    class_semantics = source_dataset.get_class_semantics()
    if unify_after_llm_enabled(cfg):
        assert_full_after_llm_semantics(
            class_semantics, num_total_classes=int(attribute.shape[0]),
            context="precompute_class_phi",
        )
    if len(class_semantics) != int(attribute.shape[0]):
        raise ValueError(
            f"Class semantic record count mismatch: got {len(class_semantics)}, "
            f"expected {int(attribute.shape[0])}.")

    # --- Encode via the exact trainer code path ---------------------------
    device = args.device
    model = HSPRN(cfg).to(device)
    model.eval()
    print(f"Encoding {len(class_semantics)} classes via psc.encode_class_phi "
          f"(itemwise={bool(getattr(model.psc, 'itemwise_pool', False))}) ...")
    phi = model.psc.encode_class_phi(class_semantics)
    phi = phi.detach().float().cpu()
    if phi.shape[0] != len(class_semantics):
        raise RuntimeError(f"encode_class_phi returned {phi.shape}, expected "
                           f"({len(class_semantics)}, D).")

    class_names = list(getattr(source_dataset, "class_names", []) or [])
    cos_before = mean_offdiag_cos(phi)
    if args.center:
        phi = phi - phi.mean(dim=0, keepdim=True)
        phi = F.normalize(phi, p=2, dim=-1)
    cos_after = mean_offdiag_cos(phi)
    output = {
        "embeddings": phi,
        "class_names": class_names,
        "class_semantics": class_semantics,
        "source": "model.psc.encode_class_phi",
        "config": os.path.abspath(args.config),
        "itemwise_pool": bool(getattr(model.psc, "itemwise_pool", False)),
        "field_norm": bool(getattr(model.psc, "field_norm", False)),
        "field_weights": getattr(model.psc, "field_weights", None),
        "centered": bool(args.center),
        "shared_dim": int(getattr(model.psc, "shared_dim", -1)),
        "export_sentences": bool(args.export_sentences),
        "control": args.control,
    }

    # v9.6 E2: per-sentence embeddings (CuPL-style ensemble input) and/or the
    # WaffleCLIP control variants. When --control is set, BOTH the class-level
    # embeddings and the sentence tensor are rebuilt from the control
    # sentences so bank/ensemble stay mutually consistent.
    if args.export_sentences or args.control != "none":
        sentences_per_class = model.psc.class_phi_sentence_texts(class_semantics)
        if args.control != "none":
            import random as _random
            rng = _random.Random(42)
            sentences_per_class = build_control_sentences(
                sentences_per_class, args.control, rng)
            phi_ctrl = []
            for sentences in sentences_per_class:
                emb = model.psc._encode_texts(list(sentences))
                phi_ctrl.append(emb.mean(dim=0).detach())
            phi = torch.stack(phi_ctrl, dim=0)
            output["embeddings"] = phi
            output["source"] = f"control:{args.control}"
            cos_after = mean_offdiag_cos(phi)
        max_s = max(len(s) for s in sentences_per_class)
        d = int(phi.shape[-1])
        sent_emb = torch.zeros(len(sentences_per_class), max_s, d)
        sent_mask = torch.zeros(len(sentences_per_class), max_s, dtype=torch.bool)
        with torch.no_grad():
            for ci, sentences in enumerate(sentences_per_class):
                emb = model.psc._encode_texts(list(sentences)).float().cpu()
                if emb.shape[0] != len(sentences):
                    raise RuntimeError(
                        f"sentence encode mismatch: {emb.shape[0]} vs {len(sentences)}")
                sent_emb[ci, :len(sentences)] = emb
                sent_mask[ci, :len(sentences)] = True
        output["embeddings_sentences"] = sent_emb
        output["sentence_mask"] = sent_mask
        output["sentence_texts"] = sentences_per_class
        print(f"Per-sentence tensor: {tuple(sent_emb.shape)} "
              f"(S_max={max_s}, classes={len(sentences_per_class)}, "
              f"control={args.control})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print(f"Saved phi_t {tuple(phi.shape)} -> {out_path}")
    print(f"  mean off-diagonal cosine: {cos_before:.4f} -> {cos_after:.4f} "
          f"(centered={bool(args.center)})")
    print(f"  norm mean={phi.norm(dim=-1).mean().item():.4f} "
          f"(encode_class_phi returns UNnormalized phi; Ws+normalize happen in-model)")


if __name__ == "__main__":
    main()
