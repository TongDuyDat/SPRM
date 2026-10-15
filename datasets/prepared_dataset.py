"""Dataset loader for self-contained packages built under prepared_datasets/."""

import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SUPPORTED_DATASETS = ("AWA2", "CUB", "SUN", "APY")


def normalize_dataset_name(dataset_name):
    """Return a canonical dataset name."""
    name = str(dataset_name).strip().upper()
    if name not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported datasets: {supported}")
    return name


def _first_present(record, *keys):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return len(value) == 0
    except TypeError:
        return False


def _canonical_semantic_record(record, index, require_full_semantics):
    if not isinstance(record, Mapping):
        raise TypeError(f"Class semantic record #{index} must be a mapping, got {type(record)}.")

    name_c = _first_present(record, "name_c", "name", "class_name", "label")
    attr_c = _first_present(record, "attr_c", "attributes", "attrs", "attr", "attribute")
    hier_c = _first_present(record, "hier_c", "hierarchy", "hier", "parents", "path")
    desc_c = _first_present(record, "desc_c", "description", "desc", "text", "caption")

    fields = {
        "name_c": name_c,
        "attr_c": attr_c,
        "hier_c": hier_c,
        "desc_c": desc_c,
    }
    missing = [key for key, value in fields.items() if _is_missing(value)]
    if require_full_semantics and missing:
        raise ValueError(
            f"Class semantic record #{index} is missing required fields: {missing}. "
            "Training requires name_c, attr_c, hier_c, and desc_c; provide them via "
            "data.semantic_metadata_path. If using an explicit semantic_embedding_path, "
            "the .pt file must also contain class_semantics for validation."
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


def load_semantic_embeddings(path):
    """Load a cached class-level semantic embedding matrix (C, D) from a .pt/.pth file."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".pt", ".pth"}:
        raise ValueError(f"Unsupported semantic embedding file extension: {ext}")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("embeddings", "semantic_embeddings", "features", "attribute", "att"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise KeyError(
                f"No embedding matrix found in {path}. Expected one of: "
                "embeddings, semantic_embeddings, features, attribute, att."
            )
    tensor = torch.as_tensor(obj, dtype=torch.float32)
    if tensor.dim() != 2:
        raise ValueError(f"Semantic embedding matrix must be 2-D, got shape {tuple(tensor.shape)}.")
    return tensor


class PreparedZSLDataset(Dataset):
    """Read raw images, splits, attributes, and semantics from one package."""

    def __init__(
        self,
        prepared_root,
        dataset_name,
        split="trainval",
        image_size=224,
        transform=None,
        require_full_semantics=False,
        semantics_override_path=None,
    ):
        self.dataset_name = normalize_dataset_name(dataset_name)
        self.split = split
        self.image_size = int(image_size)
        self.require_full_semantics = bool(require_full_semantics)
        # Optional override of class_semantics.json; the package file is untouched.
        self.semantics_override_path = (
            str(Path(semantics_override_path).expanduser())
            if semantics_override_path else None
        )

        root = Path(prepared_root).expanduser().resolve()
        package_dir = root if root.name.upper() == self.dataset_name else root / self.dataset_name
        self.package_dir = package_dir.resolve()
        manifest_path = self.package_dir / "manifest.yaml"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Prepared dataset manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
        if str(manifest.get("dataset", "")).upper() != self.dataset_name:
            raise ValueError(
                f"Prepared package dataset mismatch: expected {self.dataset_name}, "
                f"got {manifest.get('dataset')!r}."
            )
        if int(manifest.get("num_missing_images", 0)) != 0:
            raise ValueError(
                f"Prepared package {self.dataset_name} has missing images; rebuild it "
                "with --require-complete-images."
            )

        files = manifest.get("files") or {}
        metadata_dir = self.package_dir / "metadata"
        attributes_path = self._package_file(files.get("attributes"), metadata_dir / "attributes.npy")
        classes_path = self._package_file(files.get("classes"), metadata_dir / "classes.json")
        images_path = self._package_file(files.get("images"), metadata_dir / "images.jsonl")
        splits_path = self._package_file(files.get("splits"), metadata_dir / "splits.json")
        semantics_path = self._package_file(
            files.get("class_semantics") or files.get("class_descriptions"),
            metadata_dir / "class_semantics.json",
        )
        if self.semantics_override_path:
            semantics_path = Path(self.semantics_override_path)
            if not semantics_path.is_file():
                raise FileNotFoundError(
                    f"semantics_override_path not found: {semantics_path}")

        self.attribute = np.asarray(np.load(attributes_path), dtype=np.float32)
        with classes_path.open("r", encoding="utf-8") as handle:
            classes = sorted(json.load(handle), key=lambda item: int(item["class_index"]))
        self.class_names = [str(item["class_name"]) for item in classes]
        if self.attribute.shape[0] != len(self.class_names):
            raise ValueError(
                f"Class/attribute count mismatch in {self.dataset_name}: "
                f"{len(self.class_names)} classes vs {self.attribute.shape[0]} rows."
            )

        with semantics_path.open("r", encoding="utf-8") as handle:
            semantic_records = json.load(handle)
        # Accept {class_name: record} as well as a list; order by class_names.
        if isinstance(semantic_records, dict):
            semantic_records = [
                semantic_records[name] for name in self.class_names
            ]
        if len(semantic_records) != len(self.class_names):
            raise ValueError(
                f"Semantic class count mismatch in {self.dataset_name}: "
                f"{len(semantic_records)} vs {len(self.class_names)}."
            )
        self.class_semantics = [
            _canonical_semantic_record(record, index, self.require_full_semantics)
            for index, record in enumerate(semantic_records)
        ]

        with splits_path.open("r", encoding="utf-8") as handle:
            split_ids = json.load(handle)
        if split not in split_ids:
            raise ValueError(
                f"Unknown prepared split {split!r}; available: {sorted(split_ids)}"
            )

        records_by_split = {name: [] for name in split_ids}
        with images_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record_split = record.get("split")
                if record_split in records_by_split:
                    records_by_split[record_split].append(record)
        records = records_by_split[split]
        record_ids = [str(record["record_id"]) for record in records]
        expected_ids = [str(record_id) for record_id in split_ids[split]]
        if record_ids != expected_ids:
            raise ValueError(
                f"Split order mismatch between images.jsonl and splits.json for "
                f"{self.dataset_name}/{split}."
            )

        self.records = records
        self.labels = torch.tensor(
            [int(record["label"]) for record in records],
            dtype=torch.long,
        )
        self.image_paths = [self._resolve_image(record) for record in records]
        self.transform = transform or self._default_transform(split)

        seen_labels = {
            int(record["label"])
            for name in ("trainval", "test_seen")
            for record in records_by_split.get(name, [])
        }
        unseen_labels = {
            int(record["label"])
            for record in records_by_split.get("test_unseen", [])
        }
        self.seen_classes = np.asarray(sorted(seen_labels), dtype=np.int64)
        self.unseen_classes = np.asarray(sorted(unseen_labels), dtype=np.int64)

    def _package_file(self, manifest_value, fallback):
        path = self.package_dir / manifest_value if manifest_value else Path(fallback)
        path = path.resolve()
        if not path.is_relative_to(self.package_dir) or not path.is_file():
            raise FileNotFoundError(f"Prepared package file not found: {path}")
        return path

    def _resolve_image(self, record):
        if not record.get("exists", False):
            raise FileNotFoundError(
                f"Prepared image record is marked missing: {record.get('record_id')}"
            )
        relative_path = Path(str(record["package_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Prepared image path must stay relative to package root: {relative_path}"
            )
        return self.package_dir / relative_path

    def _default_transform(self, split):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        resize_size = int(round(self.image_size / 0.875))
        operations = [transforms.Resize(resize_size)]
        if split == "train":
            operations.extend([
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            operations.append(transforms.CenterCrop(self.image_size))
        operations.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return transforms.Compose(operations)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path = self.image_paths[index]
        if not path.is_file():
            raise FileNotFoundError(f"Prepared image file not found: {path}")
        image = Image.open(path).convert("RGB")
        image = self.transform(image) if self.transform is not None else image
        return image, self.labels[index]

    @property
    def num_classes(self):
        return self.attribute.shape[0]

    def get_class_semantics(self, class_indices=None):
        if class_indices is None:
            return list(self.class_semantics)
        return [self.class_semantics[int(index)] for index in class_indices]


def build_prepared_zsl_splits(
    prepared_root,
    dataset_name,
    require_full_semantics=True,
    semantic_embedding_path=None,
    **kwargs,
):
    split_kwargs = dict(kwargs)
    split_kwargs["require_full_semantics"] = require_full_semantics
    trainval = PreparedZSLDataset(prepared_root, dataset_name, split="trainval", **split_kwargs)
    test_seen = PreparedZSLDataset(prepared_root, dataset_name, split="test_seen", **split_kwargs)
    test_unseen = PreparedZSLDataset(prepared_root, dataset_name, split="test_unseen", **split_kwargs)
    semantic = torch.from_numpy(trainval.attribute)
    if semantic_embedding_path:
        semantic = load_semantic_embeddings(semantic_embedding_path)
        if semantic.shape[0] != trainval.num_classes:
            raise ValueError(
                f"Semantic embedding class count mismatch: got {semantic.shape[0]}, "
                f"expected {trainval.num_classes} for {dataset_name}."
            )
    return {
        "trainval": trainval,
        "test_seen": test_seen,
        "test_unseen": test_unseen,
        "attribute": semantic,
        "seen_classes": trainval.seen_classes,
        "unseen_classes": trainval.unseen_classes,
    }
