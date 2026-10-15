#!/usr/bin/env python
"""Precompute class text embeddings phi for LPR with the frozen CLIP ViT-B/16 text tower.

phi(t_c) is the mean of the present field vectors (name, attr, hier, desc); each field
vector is the mean of its raw, un-normalised sentence embeddings (not Eq. 1 sentence-mean).
Encoded by the same code path the model uses (``PriSematicConstructor.encode_class_phi``).

Records come from either
  * the prepared package (default when ``data.source == prepared`` and ``--semantics`` is absent), or
  * ``--semantics`` JSON: a list (class order) or a dict keyed by class name. A dict needs the
    row order from ``--names_file`` (one name per line) or ``--classes_json``
    (prepared_datasets/<DS>/metadata/classes.json).

    python tools/precompute_phi.py --dataset cub --output pre-trained/cub_phi_b16.pt
    python tools/precompute_phi.py --dataset mit --semantics <records>.json \
        --names_file <names>.txt --output pre-trained/mit-states_phi_b16.pt
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F

from model.sematics.constructor import PriSematicConstructor
from tools.utils import compose_config


def parse_args():
    p = argparse.ArgumentParser(description="Precompute class phi (CLIP ViT-B/16 text tower)")
    p.add_argument("--dataset", required=True, help="config/datasets/<dataset>.yaml (e.g. cub)")
    p.add_argument("--output", required=True, help="pre-trained/<ds>_phi_b16.pt")
    p.add_argument("--semantics", default=None, help="records JSON (list, or dict keyed by class name)")
    p.add_argument("--classes_json", default=None, help="classes.json giving the row order for a dict --semantics")
    p.add_argument("--names_file", default=None, help="one class name per line: row order for a dict --semantics")
    p.add_argument("--prepared_root", default=None)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_records(args, cfg):
    """Return (records, class_names)."""
    if args.semantics is None:
        if str(cfg.data.source).lower() != "prepared":
            raise ValueError(f"--semantics is required for data.source={cfg.data.source!r}.")
        from datasets.prepared_dataset import PreparedZSLDataset

        ds = PreparedZSLDataset(args.prepared_root or cfg.data.prepared_root, cfg.data.dataset,
                                split="trainval", require_full_semantics=True)
        return ds.get_class_semantics(), list(ds.class_names)

    with open(args.semantics, "r", encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, list):
        return records, [str(r.get("name_c") or r.get("name") or i) for i, r in enumerate(records)]
    if not isinstance(records, dict):
        raise ValueError(f"{args.semantics}: expected a JSON list or dict, got {type(records)}.")
    if args.names_file:
        with open(args.names_file, encoding="utf-8") as f:
            names = [l.strip() for l in f if l.strip()]
    elif args.classes_json:
        with open(args.classes_json, encoding="utf-8") as f:
            classes = sorted(json.load(f), key=lambda c: int(c["class_index"]))
        names = [str(c["class_name"]) for c in classes]
    else:
        raise ValueError("A dict --semantics needs --names_file or --classes_json for the row order.")
    bad = [n for n in names if n not in records or "_error" in records[n]]
    if bad:
        raise ValueError(f"{len(bad)} class(es) missing or failed in {args.semantics}, e.g. {bad[:5]}")
    return [records[n] for n in names], names


def main():
    args = parse_args()
    cfg = compose_config(args.dataset)[0]
    records, class_names = load_records(args, cfg)
    print(f"Loaded {len(records)} class records")

    enc_cfg = SimpleNamespace(model=SimpleNamespace(semantic=SimpleNamespace(
        backbone_type="clip", pretrained=True, clip_model=cfg.model.vision.clip_model,
        max_text_len=77, input_dim=None, shared_dim=512, ws_rank=32)))
    psc = PriSematicConstructor(enc_cfg).to(args.device).eval()
    phi = psc.encode_class_phi(records).detach().float().cpu()
    if phi.shape[0] != len(records):
        raise RuntimeError(f"encode_class_phi returned {tuple(phi.shape)} for {len(records)} records")

    out = {
        "embeddings": phi,
        "class_names": class_names,
        "class_semantics": records,
        "source": "model.psc.encode_class_phi",
        "itemwise_pool": True,
        "field_norm": False,
        "shared_dim": 512,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    n = F.normalize(phi, p=2, dim=-1)
    gram = n @ n.T
    off = ~torch.eye(gram.shape[0], dtype=torch.bool)
    print(f"Saved phi {tuple(phi.shape)} -> {out_path}")
    print(f"  mean off-diag cosine: {float(gram[off].mean()):.4f}")


if __name__ == "__main__":
    main()
