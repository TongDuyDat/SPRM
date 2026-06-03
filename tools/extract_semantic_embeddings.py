#!/usr/bin/env python
"""Offline semantic embedding extraction for HSPRN.

This script implements the expensive stage:
    t_c = {name_c, attr_c, hier_c, desc_c} -> text prompt -> BERT/CLIP embedding

The training code should consume the saved tensor instead of running the text
encoder inside PriSematicConstructor.

Expected metadata JSON format:
{
  "classes": [
    {
      "name": "zebra",
      "attributes": ["black", "white", "striped"],
      "hierarchy": ["animal", "mammal", "equine"],
      "description": "a black and white striped animal"
    }
  ]
}

A plain list of class records is also accepted.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from sematics.extractor import SemanticEmbeddingExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and cache class semantic embeddings")
    parser.add_argument("--metadata", type=str, required=True,
                        help="JSON file containing class-level t_c records")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pt path")
    parser.add_argument("--encoder", choices=["bert", "clip"], default="clip",
                        help="Text encoder used for offline extraction")
    parser.add_argument("--model_name", type=str, default=None,
                        help="BERT model name or CLIP model name/path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normalize", action="store_true",
                        help="L2-normalize extracted embeddings before saving")
    return parser.parse_args()


def load_class_records(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        if "classes" not in obj:
            raise KeyError("Metadata dict must contain a 'classes' key.")
        records = obj["classes"]
    elif isinstance(obj, list):
        records = obj
    else:
        raise TypeError("Metadata must be either a list of records or a dict with 'classes'.")

    if not records:
        raise ValueError("Metadata contains no class records.")
    return records


def class_names(records):
    names = []
    for i, record in enumerate(records):
        names.append(record.get("name") or record.get("class_name") or record.get("label") or str(i))
    return names


@torch.no_grad()
def encode_with_bert(prompts, model_name, max_length, batch_size, device):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    chunks = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        out = model(**encoded).last_hidden_state[:, 0, :]
        chunks.append(out.cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def encode_with_clip(prompts, model_name, max_length, batch_size, device):
    from sematics.clip import load, tokenize

    model, _ = load(model_name, device=device, jit=False)
    model.eval()

    chunks = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        tokens = tokenize(batch, context_length=max_length, truncate=True).to(device)
        out = model.encode_text(tokens).float()
        chunks.append(out.cpu())
    return torch.cat(chunks, dim=0)


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    default_model = "bert-base-uncased" if args.encoder == "bert" else "ViT-B/32"
    model_name = args.model_name or default_model
    extractor = SemanticEmbeddingExtractor(
        backend=args.encoder,
        model_name=model_name,
        max_length=args.max_length,
    )

    records = load_class_records(args.metadata)
    prompts = extractor.build_prompts(records)

    if args.encoder == "bert":
        max_length = args.max_length or 128
        embeddings = encode_with_bert(prompts, model_name, max_length, args.batch_size, device)
    else:
        max_length = args.max_length or 77
        embeddings = encode_with_clip(prompts, model_name, max_length, args.batch_size, device)

    if args.normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    output = {
        "embeddings": embeddings,
        "prompts": prompts,
        "class_names": class_names(records),
        "encoder": args.encoder,
        "model_name": model_name,
        "max_length": max_length,
        "normalized": bool(args.normalize),
        "source_metadata": os.path.abspath(args.metadata),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print(f"Saved {tuple(embeddings.shape)} semantic embeddings to {out_path}")


if __name__ == "__main__":
    main()
