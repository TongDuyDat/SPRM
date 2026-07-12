#!/usr/bin/env python
"""Prepare publishable raw-image ZSL dataset packages.

This tool bridges the xlsa17 split metadata in ``data/<DATASET>`` and raw
downloaded images in ``datasets/``. It writes explicit metadata files for the
training pipeline and can optionally organize images as symlinks or copies.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


SUPPORTED_DATASETS = ("AWA1", "AWA2", "CUB", "SUN", "APY")
SPLITS = ("trainval", "test_seen", "test_unseen")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
TOOL_METADATA_DIR = Path(__file__).resolve().parent / "dataset_metadata"


def normalize_dataset_name(dataset_name):
    name = str(dataset_name).strip().upper()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: {', '.join(SUPPORTED_DATASETS)}"
        )
    return name


def parse_dataset_list(value):
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("Expected at least one dataset name.")
    return [normalize_dataset_name(name) for name in names]


def mat_indices(indices):
    return np.asarray(indices).squeeze().astype(np.int64) - 1


def matlab_to_string(value):
    while isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.astype(str).ravel()).strip()
        if value.dtype == object and value.size:
            value = value.ravel()[0]
            continue
        if value.size == 1:
            value = value.item()
            continue
        return str(value).strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def read_indexed_text_file(path):
    path = Path(path)
    if not path.exists():
        return None
    values = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) > 1 and parts[0].isdigit():
                value = " ".join(parts[1:])
            elif "." in line and line.split(".", 1)[0].isdigit():
                value = line.split(".", 1)[1]
            else:
                value = line
            values.append(value.replace("+", " ").replace("_", " "))
    return values


def read_attribute_name_file(path):
    """Read published attribute labels without normalizing their identifiers."""
    path = Path(path)
    if not path.is_file():
        return None
    values = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            values.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
    return values


def semantic_key(value):
    return " ".join(str(value).replace("+", " ").replace("_", " ").lower().split())


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


def canonical_semantic_record(record, class_name, index, require_full):
    if not isinstance(record, Mapping):
        raise TypeError(
            f"Class semantic record #{index} ({class_name}) must be a mapping, "
            f"got {type(record)}."
        )

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
    if require_full and missing:
        raise ValueError(
            f"Class semantic record #{index} ({class_name}) is missing: {missing}"
        )

    canonical = dict(record)
    canonical.update(fields)
    return canonical


def load_semantic_records(path):
    path = Path(path)
    if not path.exists():
        return None
    ext = path.suffix.lower()
    if ext == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    elif ext in {".pt", ".pth"}:
        import torch

        obj = torch.load(path, map_location="cpu")
    else:
        return None

    if isinstance(obj, Mapping):
        for key in ("class_semantics", "classes", "records", "metadata"):
            if key in obj:
                return obj[key]
        return dict(obj)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return list(obj)
    return None


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=to_jsonable)
        f.write("\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=to_jsonable))
            f.write("\n")


def write_yaml(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def relpath_or_abs(path, roots):
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute():
        return str(path)
    path = path.resolve()
    for root in roots:
        try:
            return str(path.relative_to(Path(root).resolve()))
        except ValueError:
            continue
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_mat(path):
    import scipy.io as sio

    return sio.loadmat(path)


def load_class_names(dataset_dir, att_splits, image_root, dataset_name, num_classes):
    candidates = [
        dataset_dir / "allclasses.txt",
        dataset_dir / "classes.txt",
        dataset_dir / "Animals_with_Attributes2" / "classes.txt",
        dataset_dir / "Animals_with_Attributes" / "classes.txt",
    ]
    external_candidates = {
        "AWA1": [image_root / "Animals_with_Attributes1" / "classes.txt"],
        "AWA2": [image_root / "Animals_with_Attributes2" / "classes.txt"],
        "CUB": [image_root / "CUB_200_2011" / "classes.txt"],
    }
    candidates.extend(external_candidates.get(dataset_name, []))
    for path in candidates:
        values = read_indexed_text_file(path)
        if values and len(values) >= num_classes:
            return values[:num_classes]

    if "allclasses_names" in att_splits:
        values = [
            matlab_to_string(v).replace("+", " ").replace("_", " ")
            for v in att_splits["allclasses_names"].squeeze()
        ]
        if len(values) >= num_classes:
            return values[:num_classes]

    return [f"{dataset_name.lower()}_class_{i}" for i in range(num_classes)]


def load_attribute_names(dataset_dir, image_root, dataset_name, num_attributes):
    """Load names from the matching dataset release and verify column count."""
    dataset_name = normalize_dataset_name(dataset_name)
    text_candidates = {
        "AWA1": [
            dataset_dir / "predicates.txt",
            dataset_dir / "Animals_with_Attributes" / "predicates.txt",
            image_root / "Animals_with_Attributes1" / "predicates.txt",
            image_root / "Animals_with_Attributes1" / "Animals_with_Attributes2" / "predicates.txt",
        ],
        "AWA2": [
            dataset_dir / "predicates.txt",
            dataset_dir / "Animals_with_Attributes2" / "predicates.txt",
            image_root / "Animals_with_Attributes2" / "predicates.txt",
        ],
        "CUB": [
            dataset_dir / "attributes.txt",
            image_root / "CUB_200_2011" / "attributes" / "attributes.txt",
            image_root / "attributes.txt",
        ],
        "APY": [
            dataset_dir / "attribute_names.txt",
            TOOL_METADATA_DIR / "apy_attribute_names.txt",
        ],
        "SUN": [],
    }
    invalid_sources = []
    for path in text_candidates[dataset_name]:
        values = read_attribute_name_file(path)
        if values is None:
            continue
        if len(values) == int(num_attributes):
            return values, relpath_or_abs(path, [Path.cwd()])
        invalid_sources.append(f"{path} ({len(values)} names)")

    if dataset_name == "SUN":
        mat_candidates = [
            dataset_dir / "attributes.mat",
            image_root / "SUNAttributeDB" / "SUNAttributeDB" / "attributes.mat",
        ]
        for path in mat_candidates:
            if not path.is_file():
                continue
            data = load_mat(path)
            raw_attributes = data.get("attributes")
            if raw_attributes is None:
                invalid_sources.append(f"{path} (missing 'attributes' variable)")
                continue
            values = [matlab_to_string(value) for value in raw_attributes.squeeze()]
            if len(values) == int(num_attributes):
                return values, relpath_or_abs(path, [Path.cwd()])
            invalid_sources.append(f"{path} ({len(values)} names)")

    details = "; ".join(invalid_sources) if invalid_sources else "no source file found"
    raise FileNotFoundError(
        f"Could not load {num_attributes} published attribute names for {dataset_name}: "
        f"{details}."
    )


def image_roots(data_root, dataset_dir, dataset_name, image_root):
    roots = [
        dataset_dir,
        dataset_dir / "JPEGImages",
        dataset_dir / "images",
        dataset_dir / "Animals_with_Attributes2" / "JPEGImages",
        dataset_dir / "Animals_with_Attributes" / "JPEGImages",
        dataset_dir / "CUB_200_2011" / "images",
        data_root / dataset_name,
        data_root,
        image_root,
    ]
    dataset_roots = {
        "AWA1": [
            image_root / "Animals_with_Attributes1",
            image_root / "Animals_with_Attributes1" / "JPEGImages",
        ],
        "AWA2": [
            image_root / "Animals_with_Attributes2",
            image_root / "Animals_with_Attributes2" / "JPEGImages",
        ],
        "CUB": [
            image_root / "CUB_200_2011",
            image_root / "CUB_200_2011" / "images",
        ],
        "SUN": [
            image_root / "SUNAttributeDB",
            image_root / "SUNAttributeDB" / "images",
        ],
        "APY": [
            image_root / "images_att",
            image_root / "VOCdevkit",
            image_root / "VOCdevkit" / "VOC2008",
            image_root / "VOCdevkit" / "VOC2008" / "JPEGImages",
            image_root / "ayahoo_test_images",
        ],
    }
    roots.extend(dataset_roots.get(dataset_name, []))
    deduped = []
    seen = set()
    for root in roots:
        norm = Path(root)
        key = str(norm)
        if key not in seen:
            deduped.append(norm)
            seen.add(key)
    return deduped


def image_index_roots(data_root, dataset_dir, dataset_name, image_root):
    roots = {
        "AWA1": [
            dataset_dir / "Animals_with_Attributes" / "JPEGImages",
            dataset_dir / "Animals_with_Attributes1" / "JPEGImages",
            image_root / "Animals_with_Attributes1" / "JPEGImages",
        ],
        "AWA2": [
            dataset_dir / "Animals_with_Attributes2" / "JPEGImages",
            dataset_dir / "Animals_with_Attributes" / "JPEGImages",
            image_root / "Animals_with_Attributes2" / "JPEGImages",
        ],
        "CUB": [
            dataset_dir / "CUB_200_2011" / "images",
            image_root / "CUB_200_2011" / "images",
        ],
        "SUN": [
            dataset_dir / "images",
            image_root / "SUNAttributeDB" / "images",
        ],
        "APY": [
            dataset_dir / "images_att",
            dataset_dir / "ayahoo_test_images",
            image_root / "images_att",
            image_root / "VOCdevkit" / "VOC2008" / "JPEGImages",
            image_root / "ayahoo_test_images",
        ],
    }
    deduped = []
    seen = set()
    for root in roots.get(dataset_name, []):
        root = Path(root)
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def build_existing_image_set(roots):
    existing = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                    existing.add(str(Path(dirpath) / filename))
    return existing


def source_candidates(dataset_name, raw_path, data_root, dataset_dir, image_root):
    raw = str(raw_path).replace("\\", "/")
    parts = [part for part in raw.split("/") if part and part != "."]
    candidates = []
    seen = set()

    def add(path):
        norm = Path(path)
        key = str(norm)
        if key not in seen:
            candidates.append(norm)
            seen.add(key)

    canonical = canonical_raw_subpath(dataset_name, raw)
    add(dataset_dir / canonical)
    add(data_root / dataset_name / canonical)
    add(image_root / canonical)

    if dataset_name in {"AWA1", "AWA2"}:
        sub = marker_subpath(parts, {"Animals_with_Attributes1", "Animals_with_Attributes2", "Animals_with_Attributes"})
        if sub is not None:
            add(dataset_dir / sub)
            add(image_root / sub)

    if dataset_name == "CUB":
        sub = marker_subpath(parts, {"CUB_200_2011"}, start_at_last=True)
        if sub is not None:
            add(dataset_dir / sub)
            add(image_root / sub)

    if dataset_name == "SUN" and "images" in parts:
        tail = Path(*parts[parts.index("images"):])
        add(dataset_dir / tail)
        add(image_root / "SUNAttributeDB" / tail)

    if dataset_name == "APY":
        if "VOCdevkit" in parts:
            tail = Path(*parts[parts.index("VOCdevkit"):])
            add(dataset_dir / "images_att" / tail)
            add(image_root / tail)
        if "ayahoo_test_images" in parts:
            tail = Path(*parts[parts.index("ayahoo_test_images"):])
            add(dataset_dir / tail)
            add(image_root / tail)

    return candidates


def resolve_image_path(
    raw_path,
    roots,
    dataset_name,
    data_root,
    dataset_dir,
    image_root,
    existing_paths=None,
):
    for candidate in source_candidates(dataset_name, raw_path, data_root, dataset_dir, image_root):
        if existing_paths is not None:
            if str(candidate) in existing_paths:
                return candidate
        elif candidate.exists():
            return candidate

    if existing_paths is not None:
        return None

    raw = str(raw_path).replace("\\", "/")
    raw_candidate = Path(raw)
    if not raw_candidate.is_absolute() and raw_candidate.exists():
        return raw_candidate

    parts = [part for part in raw.split("/") if part and part != "."]
    candidates = []
    seen = set()

    def add(path):
        norm = Path(path)
        key = str(norm)
        if key not in seen:
            candidates.append(norm)
            seen.add(key)

    for root in roots:
        for start in range(max(0, len(parts) - 4), len(parts)):
            add(root.joinpath(*parts[start:]))
        if parts:
            add(root / parts[-1])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def marker_subpath(parts, markers, start_at_last=False):
    indices = [i for i, part in enumerate(parts) if part in markers]
    if not indices:
        return None
    index = indices[-1] if start_at_last else indices[0]
    return Path(*parts[index:])


def canonical_raw_subpath(dataset_name, raw_path):
    parts = [part for part in str(raw_path).replace("\\", "/").split("/") if part and part != "."]
    if dataset_name == "AWA2":
        sub = marker_subpath(parts, {"Animals_with_Attributes2", "Animals_with_Attributes"})
        if sub is not None:
            return sub
    if dataset_name == "AWA1":
        sub = marker_subpath(parts, {"Animals_with_Attributes1", "Animals_with_Attributes"})
        if sub is not None:
            return sub
    if dataset_name == "CUB":
        sub = marker_subpath(parts, {"CUB_200_2011"}, start_at_last=True)
        if sub is not None:
            return sub
    if dataset_name == "SUN":
        if "SUN" in parts:
            return Path(*parts[parts.index("SUN"):])
        if "images" in parts:
            return Path("SUN", *parts[parts.index("images"):])
    if dataset_name == "APY":
        if "APY" in parts:
            return Path(*parts[parts.index("APY"):])
        if "VOCdevkit" in parts:
            return Path("APY", "images_att", *parts[parts.index("VOCdevkit"):])
        if "ayahoo_test_images" in parts:
            return Path("APY", *parts[parts.index("ayahoo_test_images"):])
    return Path(*parts[-2:]) if len(parts) >= 2 else Path(parts[-1])


def organize_file(source, target, action, overwrite=False):
    if action == "manifest":
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            return False
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    if action == "symlink":
        os.symlink(os.path.abspath(source), target)
    elif action == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"Unknown organize action: {action}")
    return True


def semantic_metadata_path(args, dataset_name):
    if args.semantic_metadata_template:
        return Path(args.semantic_metadata_template.format(
            dataset=dataset_name,
            dataset_lower=dataset_name.lower(),
        ))
    return Path(args.semantic_metadata_root) / dataset_name / f"class_descriptions_{dataset_name.lower()}.json"


def canonical_semantics(semantic_records, class_names, require_full):
    if semantic_records is None:
        return None
    mapping = None
    if isinstance(semantic_records, Mapping):
        mapping = {semantic_key(key): value for key, value in semantic_records.items()}
    elif len(semantic_records) != len(class_names):
        raise ValueError(
            f"Semantic metadata class count mismatch: got {len(semantic_records)}, "
            f"expected {len(class_names)}."
        )

    out = []
    for index, class_name in enumerate(class_names):
        if mapping is None:
            source = semantic_records[index]
        else:
            source = mapping.get(semantic_key(class_name))
            if source is None:
                raise ValueError(f"Semantic metadata is missing class '{class_name}'.")
        out.append(canonical_semantic_record(source, class_name, index, require_full))
    return out


def generate_semantic_metadata(args, dataset_name):
    try:
        from tools import generate_descriptions as generator
    except ModuleNotFoundError:
        import generate_descriptions as generator

    dataset_dir = Path(args.data_root) / dataset_name
    att_splits = load_mat(dataset_dir / "att_splits.mat")
    class_names = load_class_names(
        dataset_dir,
        att_splits,
        Path(args.image_root),
        dataset_name,
        int(att_splits["att"].shape[1]),
    )
    output_path = semantic_metadata_path(args, dataset_name)
    generation_args = SimpleNamespace(
        api_base=args.api_base or generator.DEFAULT_API_BASE,
        api_key=args.api_key or generator.DEFAULT_API_KEY,
        model=args.model or generator.DEFAULT_MODEL,
        max_tokens=args.max_tokens or generator.DEFAULT_MAX_TOKENS,
        retries=args.retries,
        backoff=args.backoff,
        request_delay=args.request_delay,
        timeout=args.timeout,
        workers=args.description_workers,
    )
    failed = generator.generate_descriptions(generation_args, class_names, output_path)
    if failed:
        raise RuntimeError(
            f"Description generation failed for {dataset_name}: {len(failed)} classes. "
            f"Resume with the same command after checking {output_path}."
        )


def build_dataset_package(args, dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    data_root = Path(args.data_root)
    image_root = Path(args.image_root)
    dataset_dir = data_root / dataset_name
    output_dir = Path(args.output_root) / dataset_name

    res_path = dataset_dir / "res101.mat"
    att_path = dataset_dir / "att_splits.mat"
    if not res_path.exists():
        raise FileNotFoundError(f"Missing {res_path}")
    if not att_path.exists():
        raise FileNotFoundError(f"Missing {att_path}")

    res101 = load_mat(res_path)
    att_splits = load_mat(att_path)
    if "image_files" not in res101:
        raise KeyError(f"{res_path} does not contain image_files.")

    labels = res101["labels"].squeeze().astype(np.int64) - 1
    image_files = res101["image_files"].squeeze()
    attribute_raw = att_splits["att"].T.astype(np.float32)
    class_names = load_class_names(dataset_dir, att_splits, image_root, dataset_name, attribute_raw.shape[0])
    attribute_names, attribute_names_source = load_attribute_names(
        dataset_dir,
        image_root,
        dataset_name,
        attribute_raw.shape[1],
    )
    roots = image_roots(data_root, dataset_dir, dataset_name, image_root)
    existing_paths = build_existing_image_set(
        image_index_roots(data_root, dataset_dir, dataset_name, image_root)
    )

    semantic_path = semantic_metadata_path(args, dataset_name)
    semantic_records = load_semantic_records(semantic_path)
    if semantic_records is None and args.require_full_semantics:
        raise FileNotFoundError(
            f"Missing semantic metadata for {dataset_name}: {semantic_path}"
        )
    semantics = canonical_semantics(
        semantic_records,
        class_names,
        args.require_full_semantics,
    ) if semantic_records is not None else None

    split_locs = {
        "trainval": mat_indices(att_splits["trainval_loc"]),
        "test_seen": mat_indices(att_splits["test_seen_loc"]),
        "test_unseen": mat_indices(att_splits["test_unseen_loc"]),
    }

    rows = []
    missing_rows = []
    split_records = {split: [] for split in SPLITS}
    class_counts = Counter()
    organized_count = 0
    organize_jobs = {}
    source_roots = [data_root, image_root, Path.cwd()]

    for split, locs in split_locs.items():
        for split_position, global_index in enumerate(locs):
            global_index = int(global_index)
            label = int(labels[global_index])
            class_name = class_names[label]
            raw_path = matlab_to_string(image_files[global_index])
            source_path = resolve_image_path(
                raw_path,
                roots,
                dataset_name,
                data_root,
                dataset_dir,
                image_root,
                existing_paths,
            )
            canonical_relpath = Path("raw") / canonical_raw_subpath(dataset_name, raw_path)
            package_path = output_dir / canonical_relpath
            exists = source_path is not None

            if exists and not args.dry_run and args.action != "manifest":
                previous_source = organize_jobs.setdefault(package_path, source_path)
                if previous_source != source_path:
                    raise ValueError(
                        f"Package path collision for {package_path}: "
                        f"{previous_source} != {source_path}"
                    )

            row = {
                "record_id": f"{global_index:08d}",
                "dataset": dataset_name,
                "split": split,
                "split_position": int(split_position),
                "global_index": global_index,
                "matlab_index": global_index + 1,
                "label": label,
                "class_name": class_name,
                "raw_image_file": raw_path,
                "source_path": relpath_or_abs(source_path, source_roots) if source_path else None,
                "package_path": str(canonical_relpath),
                "exists": bool(exists),
                "file_size": int(source_path.stat().st_size)
                if exists and args.include_file_size
                else None,
            }
            rows.append(row)
            split_records[split].append(row["record_id"])
            class_counts[class_name] += 1
            if not exists:
                missing_rows.append(row)

    classes = [
        {
            "class_index": index,
            "class_name": name,
            "attribute": attribute_raw[index].astype(float).tolist(),
            "record_count": int(class_counts.get(name, 0)),
        }
        for index, name in enumerate(class_names)
    ]

    if args.require_complete_images and missing_rows:
        first = missing_rows[0]
        raise FileNotFoundError(
            f"{dataset_name} has {len(missing_rows)} missing images. "
            f"First missing: {first['raw_image_file']}"
        )

    if organize_jobs:
        with ThreadPoolExecutor(max_workers=args.copy_workers) as executor:
            futures = [
                executor.submit(
                    organize_file,
                    source,
                    target,
                    args.action,
                    args.overwrite,
                )
                for target, source in organize_jobs.items()
            ]
            for future in as_completed(futures):
                organized_count += int(future.result())

    summary = {
        "schema_version": 1,
        "dataset": dataset_name,
        "data_root": str(data_root),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "action": args.action,
        "num_classes": len(class_names),
        "num_records": len(rows),
        "num_missing_images": len(missing_rows),
        "num_organized_files": organized_count,
        "splits": {split: len(ids) for split, ids in split_records.items()},
        "semantic_metadata_path": str(semantic_path) if semantic_path.exists() else None,
        "has_semantic_metadata": semantics is not None,
        "attribute_names_source": attribute_names_source,
        "files": {
            "manifest": "manifest.yaml",
            "classes": "metadata/classes.json",
            "attributes": "metadata/attributes.npy",
            "attribute_names": "metadata/attribute_names.json",
            "splits": "metadata/splits.json",
            "images": "metadata/images.jsonl",
            "missing_images": "metadata/missing_images.jsonl",
            "class_semantics": "metadata/class_semantics.json" if semantics is not None else None,
            "class_descriptions": (
                f"metadata/class_descriptions_{dataset_name.lower()}.json"
                if semantics is not None else None
            ),
        },
    }

    if not args.dry_run:
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(output_dir / "manifest.yaml", summary)
        write_json(metadata_dir / "classes.json", classes)
        write_json(metadata_dir / "attribute_names.json", attribute_names or [])
        write_json(metadata_dir / "splits.json", split_records)
        write_jsonl(metadata_dir / "images.jsonl", rows)
        write_jsonl(metadata_dir / "missing_images.jsonl", missing_rows)
        np.save(metadata_dir / "attributes.npy", attribute_raw)
        if semantics is not None:
            write_json(metadata_dir / "class_semantics.json", semantics)
            write_json(
                metadata_dir / f"class_descriptions_{dataset_name.lower()}.json",
                semantics,
            )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare canonical ZSL dataset metadata and optional image packages."
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--image-root", default="./datasets")
    parser.add_argument("--output-root", default="./prepared_datasets")
    parser.add_argument("--datasets", type=parse_dataset_list, default=list(SUPPORTED_DATASETS))
    parser.add_argument(
        "--action",
        choices=["manifest", "symlink", "copy"],
        default="manifest",
        help="manifest writes metadata only; symlink/copy also organizes images.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-complete-images", action="store_true")
    parser.add_argument("--require-full-semantics", action="store_true")
    parser.add_argument("--include-file-size", action="store_true")
    parser.add_argument("--semantic-metadata-root", default="./data_semantics")
    parser.add_argument(
        "--semantic-metadata-template",
        default=None,
        help="Format string, e.g. './data_semantics/{dataset}/class_descriptions_{dataset_lower}.json'.",
    )
    parser.add_argument("--generate-descriptions", action="store_true")
    parser.add_argument("--description-workers", type=int, default=2)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=30.0)
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.description_workers < 1:
        parser.error("--description-workers must be at least 1.")
    if args.copy_workers < 1:
        parser.error("--copy-workers must be at least 1.")
    return args


def main():
    args = parse_args()
    summaries = []
    for dataset_name in args.datasets:
        if args.generate_descriptions:
            generate_semantic_metadata(args, dataset_name)
        summary = build_dataset_package(args, dataset_name)
        summaries.append(summary)
        print(
            f"{dataset_name}: records={summary['num_records']} "
            f"missing={summary['num_missing_images']} action={summary['action']}"
        )

    if not args.dry_run:
        output_root = Path(args.output_root)
        write_yaml(
            output_root / "manifest.yaml",
            {
                "schema_version": 1,
                "datasets": summaries,
            },
        )


if __name__ == "__main__":
    main()
