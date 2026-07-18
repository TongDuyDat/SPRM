"""Dataset loader for self-contained packages built under prepared_datasets/."""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .zsl_dataset import _canonical_semantic_record, normalize_dataset_name


class PreparedZSLDataset(Dataset):
    """Read raw images, splits, attributes, and semantics from one package."""

    def __init__(
        self,
        prepared_root,
        dataset_name,
        split="trainval",
        image_size=224,
        image_preprocess="imagenet",
        transform=None,
        return_semantics=False,
        require_full_semantics=False,
    ):
        self.dataset_name = normalize_dataset_name(dataset_name)
        self.split = split
        self.mode = "image"
        self.image_size = int(image_size)
        self.image_preprocess = image_preprocess
        self.return_semantics = bool(return_semantics)
        self.require_full_semantics = bool(require_full_semantics)

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
        attribute_names_path = self._package_file(
            files.get("attribute_names"), metadata_dir / "attribute_names.json"
        )
        classes_path = self._package_file(files.get("classes"), metadata_dir / "classes.json")
        images_path = self._package_file(files.get("images"), metadata_dir / "images.jsonl")
        splits_path = self._package_file(files.get("splits"), metadata_dir / "splits.json")
        semantics_path = self._package_file(
            files.get("class_semantics") or files.get("class_descriptions"),
            metadata_dir / "class_semantics.json",
        )

        self.attribute = np.asarray(np.load(attributes_path), dtype=np.float32)
        with attribute_names_path.open("r", encoding="utf-8") as handle:
            self.attribute_names = [str(value) for value in json.load(handle)]
        if len(self.attribute_names) != self.attribute.shape[1]:
            raise ValueError(
                f"Attribute-name count mismatch in {self.dataset_name}: "
                f"{len(self.attribute_names)} names vs {self.attribute.shape[1]} columns."
            )
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
        self.raw_image_paths = [str(path) for path in self.image_paths]
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
        unique_labels = sorted({int(label) for label in self.labels.tolist()})
        self.label_to_idx = {label: index for index, label in enumerate(unique_labels)}
        self.idx_to_label = {index: label for label, index in self.label_to_idx.items()}
        self.num_classes_in_split = len(unique_labels)

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
        if self.image_preprocess == "clip":
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

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
        label = self.labels[index]
        if self.return_semantics:
            return image, label, self.class_semantics[int(label)]
        return image, label

    @property
    def attr_dim(self):
        return self.attribute.shape[1]

    @property
    def num_classes(self):
        return self.attribute.shape[0]

    def get_class_attributes(self, class_indices=None):
        if class_indices is None:
            return torch.from_numpy(self.attribute)
        return torch.from_numpy(self.attribute[class_indices])

    def get_class_semantics(self, class_indices=None):
        if class_indices is None:
            return list(self.class_semantics)
        return [self.class_semantics[int(index)] for index in class_indices]


def build_prepared_zsl_dataset(prepared_root, dataset_name, split="trainval", **kwargs):
    return PreparedZSLDataset(
        prepared_root,
        dataset_name,
        split=split,
        **kwargs,
    )


def build_prepared_zsl_splits(
    prepared_root,
    dataset_name,
    require_full_semantics=True,
    **kwargs,
):
    split_kwargs = dict(kwargs)
    split_kwargs["require_full_semantics"] = require_full_semantics
    trainval = build_prepared_zsl_dataset(
        prepared_root, dataset_name, split="trainval", **split_kwargs
    )
    test_seen = build_prepared_zsl_dataset(
        prepared_root, dataset_name, split="test_seen", **split_kwargs
    )
    test_unseen = build_prepared_zsl_dataset(
        prepared_root, dataset_name, split="test_unseen", **split_kwargs
    )
    semantic = torch.from_numpy(trainval.attribute)
    return {
        "trainval": trainval,
        "test_seen": test_seen,
        "test_unseen": test_unseen,
        "attribute": semantic,
        "semantic_embedding": semantic,
        "seen_classes": trainval.seen_classes,
        "unseen_classes": trainval.unseen_classes,
    }
