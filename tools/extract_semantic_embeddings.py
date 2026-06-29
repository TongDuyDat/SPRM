#!/usr/bin/env python
"""Offline semantic embedding extraction for HSPRN.

This script implements the expensive stage:
    t_c = {name_c, attr_c, hier_c, desc_c} -> BERT/CLIP embedding

Each field is encoded as a separate text sample; fields are not concatenated
into one prompt. The saved `embeddings` tensor is the unweighted class-level
mean of the four field embeddings because the current HSPRN semantic bank
expects one phi(t_c) vector per class.

Expected metadata JSON format:
{
  "classes": [
    {
      "name_c": "zebra",
      "attr_c": ["black", "white", "striped"],
      "hier_c": ["animal", "mammal", "equine"],
      "desc_c": "a black and white striped animal"
    }
  ]
}

A plain list of class records and the `class_descriptions_<dataset>.json`
format produced by tools/generate_descriptions.py are also accepted.
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

def first_present(record, *keys):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return len(value) == 0
    except TypeError:
        return False


def canonical_record(record, index):
    name_c = first_present(record, "name_c", "name", "class_name", "label")
    attr_c = first_present(record, "attr_c", "attributes", "attrs", "attr", "attribute")
    hier_c = first_present(record, "hier_c", "hierarchy", "hier", "parents", "path")
    desc_c = first_present(record, "desc_c", "description", "desc", "text", "caption")
    fields = {
        "name_c": name_c,
        "attr_c": attr_c,
        "hier_c": hier_c,
        "desc_c": desc_c,
    }
    missing = [key for key, value in fields.items() if is_missing(value)]
    if missing:
        raise ValueError(
            f"Metadata record #{index} is missing required fields: {missing}. "
            "Each class must provide name_c, attr_c, hier_c, and desc_c."
        )

    canonical = dict(record)
    canonical.update(fields)
    canonical.setdefault("name", name_c)
    canonical.setdefault("attributes", attr_c)
    canonical.setdefault("attr", attr_c)
    canonical.setdefault("hierarchy", hier_c)
    canonical.setdefault("hier", hier_c)
    canonical.setdefault("description", desc_c)
    canonical.setdefault("desc", desc_c)
    return canonical


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
        if "classes" in obj:
            records = obj["classes"]
        elif all(isinstance(value, dict) for value in obj.values()):
            records = []
            for class_name, record in obj.items():
                record = dict(record)
                record.setdefault("name_c", class_name)
                records.append(record)
        else:
            raise KeyError(
                "Metadata dict must contain a 'classes' key or be keyed by class name."
            )
    elif isinstance(obj, list):
        records = obj
    else:
        raise TypeError("Metadata must be either a list of records or a dict with 'classes'.")

    if not records:
        raise ValueError("Metadata contains no class records.")
    return [canonical_record(record, index) for index, record in enumerate(records)]


def class_names(records):
    names = []
    for i, record in enumerate(records):
        names.append(
            record.get("name_c") or record.get("name") or
            record.get("class_name") or record.get("label") or str(i)
        )
    return names


def field_to_text(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {field_to_text(item)}" for key, item in value.items()
            if not is_missing(item)
        )
    return str(value).strip()


def field_texts(records):
    fields = ("name_c", "attr_c", "hier_c", "desc_c")
    texts = []
    for record in records:
        for field in fields:
            text = field_to_text(record[field])
            if not text:
                raise ValueError(f"Field {field} for {record.get('name_c')} is empty.")
            texts.append(text)
    return fields, texts


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
    records = load_class_records(args.metadata)
    field_names, texts = field_texts(records)

    if args.encoder == "bert":
        max_length = args.max_length or 128
        flat_embeddings = encode_with_bert(texts, model_name, max_length, args.batch_size, device)
    else:
        max_length = args.max_length or 77
        flat_embeddings = encode_with_clip(texts, model_name, max_length, args.batch_size, device)

    field_embeddings = flat_embeddings.view(len(records), len(field_names), -1)
    embeddings = field_embeddings.mean(dim=1)
    if args.normalize:
        field_embeddings = F.normalize(field_embeddings, p=2, dim=-1)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    output = {
        "embeddings": embeddings,
        "field_embeddings": field_embeddings,
        "field_names": list(field_names),
        "field_texts": texts,
        "class_names": class_names(records),
        "class_semantics": records,
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
