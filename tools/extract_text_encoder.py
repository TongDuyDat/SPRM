#!/usr/bin/env python3
"""Extract minimal text-only artifacts from TinyCLIP or Sentence-BERT."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("tinyclip", "sbert"))
    parser.add_argument("--source", required=True, help="Hugging Face repo id or local path")
    parser.add_argument("--output", required=True, help="Output directory for text-only files")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer, CLIPTextModelWithProjection

    tokenizer = AutoTokenizer.from_pretrained(args.source)
    if args.backend == "tinyclip":
        model = CLIPTextModelWithProjection.from_pretrained(args.source)
        out_features = int(model.config.projection_dim)
        pooling = "clip_eos_projection"
        if any(name.startswith("vision") for name, _ in model.named_parameters()):
            raise RuntimeError("TinyCLIP extraction unexpectedly retained visual parameters.")
    else:
        model = AutoModel.from_pretrained(args.source)
        out_features = int(model.config.hidden_size)
        pooling = "attention_mask_mean"

    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    metadata = {
        "backend": args.backend,
        "source": args.source,
        "out_features": out_features,
        "pooling": pooling,
        "parameters": sum(param.numel() for param in model.parameters()),
    }
    (output / "text_encoder_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
