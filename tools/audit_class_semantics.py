#!/usr/bin/env python
"""Audit after-LLM class semantics completeness and CLIP token budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.clip_token_budget import DEFAULT_CONTEXT_LENGTH, count_tokens, get_clip_tokenizer
from tools.semantic_policy import after_llm_path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_key(value: str) -> str:
    return " ".join(str(value).replace("+", " ").replace("_", " ").lower().split())


def as_items(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        return lines if lines else ([value.strip()] if value.strip() else [])
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def load_class_names(args) -> list[str]:
    if args.classes_file:
        names = []
        with open(args.classes_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) > 1 and parts[0].isdigit():
                    line = " ".join(parts[1:])
                names.append(line.replace("+", " ").replace("_", " ").strip())
        return names

    if args.dataset:
        # Prefer prepared package classes if present.
        prepared = Path(args.prepared_root) / str(args.dataset).upper() / "metadata" / "classes.json"
        if prepared.exists():
            classes = load_json(prepared)
            if isinstance(classes, list):
                names = []
                for c in classes:
                    if isinstance(c, dict):
                        raw = c.get("class_name") or c.get("name") or c.get("label") or ""
                    else:
                        raw = c
                    names.append(str(raw).replace("+", " ").replace("_", " ").strip())
                if names and all(names):
                    return names
        # XLSA17 allclasses_names
        import scipy.io as sio

        mat_path = Path(args.data_root) / str(args.dataset).upper() / "att_splits.mat"
        data = sio.loadmat(mat_path)
        names = []
        for value in data["allclasses_names"].squeeze():
            while hasattr(value, "size") and getattr(value, "size", 0) == 1:
                value = value.item()
            names.append(str(value).replace("+", " ").replace("_", " ").strip())
        return names

    raise SystemExit("Provide --dataset or --classes_file.")


def resolve_semantics_path(args) -> Path:
    if args.semantics:
        return Path(args.semantics)
    if not args.dataset:
        raise SystemExit("--semantics or --dataset is required.")
    return Path(after_llm_path(args.semantic_root, args.dataset))


def audit(args) -> int:
    class_names = load_class_names(args)
    path = resolve_semantics_path(args)
    if not path.exists():
        print(f"ERROR: missing semantics file: {path}", file=sys.stderr)
        return 2

    raw = load_json(path)
    if isinstance(raw, list):
        mapping = {
            normalize_key(rec.get("name_c") or rec.get("name") or ""): rec
            for rec in raw
            if isinstance(rec, dict)
        }
    elif isinstance(raw, dict):
        mapping = {normalize_key(k): v for k, v in raw.items()}
    else:
        print(f"ERROR: unsupported JSON type in {path}", file=sys.stderr)
        return 2

    tokenizer = get_clip_tokenizer() if not args.skip_token_check else None
    missing_classes = []
    empty_fields = []
    token_counts: list[int] = []
    overflow_items = []

    field_keys = ("attr_c", "hier_c", "desc_c")
    for class_name in class_names:
        rec = mapping.get(normalize_key(class_name))
        if rec is None or not isinstance(rec, dict):
            missing_classes.append(class_name)
            continue
        name_c = rec.get("name_c") or rec.get("name") or class_name
        if not str(name_c).strip():
            empty_fields.append((class_name, "name_c"))
        for field in field_keys:
            items = as_items(rec.get(field))
            if not items:
                empty_fields.append((class_name, field))
                continue
            if tokenizer is None:
                continue
            for item in items:
                n = count_tokens(item, tokenizer=tokenizer)
                token_counts.append(n)
                if n > args.max_tokens:
                    overflow_items.append(
                        {"class": class_name, "field": field, "tokens": n, "text": item[:160]}
                    )

    report = {
        "semantics_path": str(path),
        "num_classes_expected": len(class_names),
        "num_classes_present": len(class_names) - len(missing_classes),
        "missing_classes": missing_classes,
        "empty_fields": [{"class": c, "field": f} for c, f in empty_fields],
        "token_stats": {
            "count": len(token_counts),
            "min": min(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
            "mean": (sum(token_counts) / len(token_counts)) if token_counts else 0.0,
            "overflow": len(overflow_items),
            "max_tokens": args.max_tokens,
        },
        "overflow_items_sample": overflow_items[:20],
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Wrote report: {out}")

    print(json.dumps({k: report[k] for k in report if k != "overflow_items_sample"}, indent=2))
    if overflow_items:
        print(f"overflow sample (first {min(5, len(overflow_items))}):")
        for row in overflow_items[:5]:
            print(f"  {row['class']}/{row['field']}: {row['tokens']} tokens")

    failed = bool(missing_classes or empty_fields)
    if args.fail_on_overflow and overflow_items:
        failed = True
    return 1 if failed else 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit after-LLM class semantics (completeness + CLIP tokens)."
    )
    parser.add_argument("--dataset", default=None, help="Dataset name, e.g. CUB or APY.")
    parser.add_argument(
        "--semantics",
        default=None,
        help="Path to after-LLM JSON. Default: data_semantics/<DS>/class_descriptions_<ds>_after_llm.json",
    )
    parser.add_argument("--semantic-root", default="./data_semantics")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--prepared-root", default="./prepared_datasets")
    parser.add_argument("--classes_file", default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--skip-token-check", action="store_true")
    parser.add_argument(
        "--fail-on-overflow",
        action="store_true",
        help="Exit non-zero if any item exceeds max tokens.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main():
    args = parse_args()
    raise SystemExit(audit(args))


if __name__ == "__main__":
    main()
