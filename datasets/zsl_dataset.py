"""
ZSL Dataset — loads xlsa17 Proposed Split .mat files.

Expected directory structure under data_root:
    data_root/
        AWA1/   (or AWA2, CUB, SUN, APY)
            res101.mat        — features + labels
            att_splits.mat    — attributes + seen/unseen split indices

The .mat files follow the format from:
    Xian et al., "Zero-Shot Learning — A Comprehensive Evaluation of the
    Good, the Bad and the Ugly", TPAMI 2018.

Download: https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip
"""

import json
import os
from collections.abc import Mapping, Sequence

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


SUPPORTED_DATASETS = ("AWA1", "AWA2", "CUB", "SUN", "APY")


def normalize_dataset_name(dataset_name):
    """Return a canonical xlsa17 dataset name."""
    name = str(dataset_name).strip().upper()
    if name not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported datasets: {supported}")
    return name


def _mat_indices(indices):
    """Convert MATLAB 1-indexed split arrays to NumPy 0-indexed int arrays."""
    return np.asarray(indices).squeeze().astype(np.int64) - 1


def _matlab_to_string(value):
    """Convert common scipy.io MATLAB string cells to a Python string."""
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


def _read_indexed_text_file(path):
    """Read files such as '1 antelope' or '134.Cape_Glossy_Starling'."""
    if not os.path.exists(path):
        return None
    values = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
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


def load_semantic_records(path):
    """Load class-level t_c records from JSON or a .pt extraction output."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    elif ext in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
    else:
        return None

    if isinstance(obj, Mapping):
        for key in ("class_semantics", "classes", "records", "metadata"):
            if key in obj:
                obj = obj[key]
                break
        else:
            return dict(obj)

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return list(obj)
    return None


def _semantic_key(value):
    return " ".join(str(value).replace("+", " ").replace("_", " ").lower().split())


def _resolve_image_path(raw_path, roots):
    """Resolve XLSA17 image_files entries against likely local image roots."""
    raw = str(raw_path).replace("\\", "/")
    if os.path.exists(raw):
        return raw

    parts = [part for part in raw.split("/") if part and part != "."]
    candidates = []
    seen = set()

    def add_candidate(path):
        norm = os.path.normpath(path)
        if norm not in seen:
            candidates.append(norm)
            seen.add(norm)

    for root in roots:
        if not root:
            continue
        add_candidate(os.path.join(root, raw))
        for start in range(len(parts)):
            add_candidate(os.path.join(root, *parts[start:]))
        if parts:
            add_candidate(os.path.join(root, parts[-1]))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _image_roots(data_root, dataset_dir, dataset_name, image_root=None):
    data_root_abs = os.path.abspath(data_root)
    dataset_dir_abs = os.path.abspath(dataset_dir)
    workspace_dataset_root = os.path.join(os.path.dirname(data_root_abs), "datasets")
    image_root_abs = os.path.abspath(image_root) if image_root else workspace_dataset_root

    roots = [
        dataset_dir,
        dataset_dir_abs,
        os.path.join(dataset_dir, "JPEGImages"),
        os.path.join(dataset_dir, "images"),
        os.path.join(dataset_dir, "Animals_with_Attributes2", "JPEGImages"),
        os.path.join(dataset_dir, "Animals_with_Attributes", "JPEGImages"),
        os.path.join(dataset_dir, "CUB_200_2011", "images"),
        os.path.join(data_root, dataset_name),
        data_root_abs,
        image_root_abs,
    ]
    dataset_roots = {
        "AWA1": [
            os.path.join(image_root_abs, "Animals_with_Attributes1"),
            os.path.join(image_root_abs, "Animals_with_Attributes1", "JPEGImages"),
        ],
        "AWA2": [
            os.path.join(image_root_abs, "Animals_with_Attributes2"),
            os.path.join(image_root_abs, "Animals_with_Attributes2", "JPEGImages"),
        ],
        "CUB": [
            os.path.join(image_root_abs, "CUB_200_2011"),
            os.path.join(image_root_abs, "CUB_200_2011", "images"),
        ],
        "SUN": [
            os.path.join(image_root_abs, "SUNAttributeDB"),
            os.path.join(image_root_abs, "SUNAttributeDB", "images"),
        ],
        "APY": [
            os.path.join(image_root_abs, "images_att"),
            os.path.join(image_root_abs, "VOCdevkit"),
            os.path.join(image_root_abs, "VOCdevkit", "VOC2008"),
            os.path.join(image_root_abs, "VOCdevkit", "VOC2008", "JPEGImages"),
            os.path.join(image_root_abs, "ayahoo_test_images"),
        ],
    }
    roots.extend(dataset_roots.get(dataset_name, []))

    resolved = []
    seen = set()
    for root in roots:
        if not root:
            continue
        norm = os.path.normpath(root)
        if norm not in seen:
            resolved.append(norm)
            seen.add(norm)
    return resolved


def load_semantic_embeddings(path):
    """Load a cached class-level semantic embedding matrix.

    Expected shape is (num_classes, semantic_dim). The recommended format is
    a .pt file produced by tools/extract_semantic_embeddings.py with an
    `embeddings` tensor. NumPy arrays and simple .mat files are accepted for
    compatibility with external ZSL embeddings.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in {".pt", ".pth"}:
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
    elif ext == ".npy":
        tensor = torch.from_numpy(np.load(path)).float()
    elif ext == ".npz":
        obj = np.load(path)
        for key in ("embeddings", "semantic_embeddings", "features", "attribute", "att"):
            if key in obj:
                tensor = torch.from_numpy(obj[key]).float()
                break
        else:
            raise KeyError(
                f"No embedding matrix found in {path}. Expected one of: "
                "embeddings, semantic_embeddings, features, attribute, att."
            )
    elif ext == ".mat":
        obj = sio.loadmat(path)
        selected_key = None
        for key in ("embeddings", "semantic_embeddings", "features", "attribute", "att"):
            if key in obj:
                selected_key = key
                break
        if selected_key is None:
            raise KeyError(
                f"No embedding matrix found in {path}. Expected one of: "
                "embeddings, semantic_embeddings, features, attribute, att."
            )
        arr = obj[selected_key]
        if selected_key == "att":
            arr = arr.T
        tensor = torch.from_numpy(arr).float()
    else:
        raise ValueError(f"Unsupported semantic embedding file extension: {ext}")

    if tensor.dim() != 2:
        raise ValueError(
            f"Semantic embeddings must have shape (num_classes, dim), got {tuple(tensor.shape)}"
        )
    return tensor


class ZSLDataset(Dataset):
    """Unified ZSL dataset for AWA1/AWA2/CUB/SUN/APY.

    Loads pre-extracted ResNet-101 features + class attributes from xlsa17
    Proposed Split (PS), OR loads raw images for end-to-end training.
    """

    def __init__(
        self,
        data_root,
        dataset_name,
        split="train",
        mode="feature",
        transform=None,
        image_size=224,
        image_preprocess="imagenet",
        return_semantics=False,
        semantic_records=None,
        semantic_metadata_path=None,
        require_full_semantics=False,
        image_root=None,
    ):
        """
        Args:
            data_root: Root directory containing dataset folders.
            dataset_name: One of 'AWA1', 'AWA2', 'CUB', 'SUN', 'APY'.
            split: 'train' | 'val_seen' | 'val_unseen' | 'test_seen' | 'test_unseen'.
            mode: 'feature' to load pre-extracted 2048-dim ResNet-101 features,
                  'image' to load raw images for backbone extraction.
            transform: Optional torchvision transform to apply to images.
            image_size: Spatial input size for the image transform.
            image_preprocess: 'imagenet' or 'clip' normalization statistics.
            return_semantics: If True, __getitem__ also returns the class t_c record.
            semantic_records: Optional list of class-level t_c records.
            semantic_metadata_path: Optional JSON/.pt path containing class-level t_c records.
            require_full_semantics: If True, require name_c/attr_c/hier_c/desc_c for each class.
            image_root: Optional root containing raw image datasets downloaded separately
                from the xlsa17 .mat files, e.g. './datasets'.
        """
        super().__init__()
        self.dataset_name = normalize_dataset_name(dataset_name)
        self.split = split
        self.mode = mode.lower()
        self.transform = transform
        self.image_size = int(image_size)
        self.image_preprocess = image_preprocess
        self.return_semantics = bool(return_semantics)
        self.require_full_semantics = bool(require_full_semantics)
        self.image_root = image_root

        dataset_dir = os.path.join(data_root, self.dataset_name)
        res_path = os.path.join(dataset_dir, "res101.mat")
        att_path = os.path.join(dataset_dir, "att_splits.mat")

        if not os.path.exists(res_path):
            raise FileNotFoundError(
                f"res101.mat not found at {res_path}. "
                f"Download xlsa17.zip from https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"
            )
        if not os.path.exists(att_path):
            raise FileNotFoundError(
                f"att_splits.mat not found at {att_path}. "
                f"Download xlsa17.zip from https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"
            )

        # ------------------------------------------------------------------
        # Load .mat files
        # ------------------------------------------------------------------
        res101 = sio.loadmat(res_path)
        att_splits = sio.loadmat(att_path)

        # Features: (num_images, 2048)
        all_features = res101["features"].T.astype(np.float32)  # .mat stores as (2048, N)
        # Labels: (num_images,) — 1-indexed in .mat → convert to 0-indexed
        all_labels = res101["labels"].squeeze().astype(np.int64) - 1

        # Attributes: (num_classes, attr_dim)
        attribute_raw = att_splits["att"].T.astype(np.float32)  # .mat stores as (attr_dim, C)
        self.attribute = attribute_raw / (np.linalg.norm(attribute_raw, axis=1, keepdims=True) + 1e-8)
        self.class_names = self._load_class_names(dataset_dir, att_splits, self.attribute.shape[0])
        self.attribute_names = self._load_attribute_names(dataset_dir)
        if semantic_records is None and semantic_metadata_path:
            semantic_records = load_semantic_records(semantic_metadata_path)
        self.class_semantics = self._build_class_semantics(attribute_raw, semantic_records)

        # ------------------------------------------------------------------
        # Proposed Split indices (1-indexed → 0-indexed)
        # ------------------------------------------------------------------
        trainval_loc = _mat_indices(att_splits["trainval_loc"])
        test_seen_loc = _mat_indices(att_splits["test_seen_loc"])
        test_unseen_loc = _mat_indices(att_splits["test_unseen_loc"])

        # Train/val split from trainval: use 80/20 for validation
        train_loc = att_splits.get("train_loc", None)
        val_loc = att_splits.get("val_loc", None)

        if train_loc is not None:
            train_loc = _mat_indices(train_loc)
        if val_loc is not None:
            val_loc = _mat_indices(val_loc)

        # Seen / unseen class indices
        self.seen_classes = np.unique(all_labels[trainval_loc])
        self.unseen_classes = np.unique(all_labels[test_unseen_loc])
        self.all_classes = np.arange(self.attribute.shape[0])

        # ------------------------------------------------------------------
        # Select split
        # ------------------------------------------------------------------
        if split == "train":
            if train_loc is not None:
                loc = train_loc
            else:
                loc = trainval_loc
        elif split == "trainval":
            loc = trainval_loc
        elif split == "val_seen":
            if val_loc is not None:
                loc = val_loc
            else:
                # fallback: use test_seen as val_seen
                loc = test_seen_loc
        elif split == "val_unseen":
            loc = test_unseen_loc
        elif split == "test_seen":
            loc = test_seen_loc
        elif split == "test_unseen":
            loc = test_unseen_loc
        else:
            raise ValueError(f"Unknown split '{split}'. Use: train/trainval/val_seen/val_unseen/test_seen/test_unseen")

        self.features = torch.from_numpy(all_features[loc])
        self.labels = torch.from_numpy(all_labels[loc])

        # ------------------------------------------------------------------
        # Setup Image Mode
        # ------------------------------------------------------------------
        if self.mode == "image":
            # Set up default transforms if none provided
            if self.transform is None:
                self.transform = self._default_transform(split)

            if "image_files" not in res101:
                raise KeyError("res101.mat does not contain image_files; raw-image mode requires it.")

            raw_paths = res101["image_files"].squeeze()
            loc_raw_paths = raw_paths[loc]
            self.raw_image_paths = [_matlab_to_string(p) for p in loc_raw_paths]
            self.image_roots = _image_roots(data_root, dataset_dir, self.dataset_name, image_root)
            self.image_paths = list(self.raw_image_paths)
            self._resolved_image_paths = {}

        # Remap labels to contiguous seen-class indices for training
        self._build_label_maps()

    def _build_label_maps(self):
        """Build mappings between original labels and contiguous indices."""
        unique_labels = torch.unique(self.labels).numpy()
        self.label_to_idx = {int(l): i for i, l in enumerate(unique_labels)}
        self.idx_to_label = {i: int(l) for i, l in enumerate(unique_labels)}
        self.num_classes_in_split = len(unique_labels)

    def _default_transform(self, split):
        if self.image_preprocess == "clip":
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        resize_size = int(round(self.image_size / 0.875))
        ops = [transforms.Resize(resize_size)]
        if split == "train":
            ops.extend([
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            ops.append(transforms.CenterCrop(self.image_size))
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return transforms.Compose(ops)

    def _load_class_names(self, dataset_dir, att_splits, num_classes):
        candidates = [
            os.path.join(dataset_dir, "allclasses.txt"),
            os.path.join(dataset_dir, "classes.txt"),
            os.path.join(dataset_dir, "Animals_with_Attributes2", "classes.txt"),
            os.path.join(dataset_dir, "Animals_with_Attributes", "classes.txt"),
        ]
        for path in candidates:
            values = _read_indexed_text_file(path)
            if values and len(values) >= num_classes:
                return values[:num_classes]

        if "allclasses_names" in att_splits:
            values = [_matlab_to_string(v).replace("+", " ").replace("_", " ")
                      for v in att_splits["allclasses_names"].squeeze()]
            if len(values) >= num_classes:
                return values[:num_classes]

        return [f"class_{i}" for i in range(num_classes)]

    def _load_attribute_names(self, dataset_dir):
        candidates = [
            os.path.join(dataset_dir, "predicates.txt"),
            os.path.join(dataset_dir, "attributes.txt"),
            os.path.join(dataset_dir, "Animals_with_Attributes2", "predicates.txt"),
            os.path.join(dataset_dir, "Animals_with_Attributes", "predicates.txt"),
        ]
        for path in candidates:
            values = _read_indexed_text_file(path)
            if values:
                return values
        return None

    def _format_attributes(self, attr_values):
        if not self.attribute_names:
            return attr_values.astype(float).tolist()
        pairs = []
        for value, name in zip(attr_values, self.attribute_names):
            if float(value) > 0:
                pairs.append(name)
        return pairs

    def _build_class_semantics(self, attribute_raw, semantic_records=None):
        semantic_mapping = None
        if isinstance(semantic_records, Mapping):
            semantic_mapping = {
                _semantic_key(key): value for key, value in semantic_records.items()
            }
        elif semantic_records is not None and len(semantic_records) != len(self.class_names):
            raise ValueError(
                f"Semantic metadata class count mismatch: got {len(semantic_records)}, "
                f"expected {len(self.class_names)} for {self.dataset_name}."
            )
        if semantic_records is None and self.require_full_semantics:
            raise ValueError(
                "Training requires full class semantics with name_c, attr_c, hier_c, and desc_c. "
                "Provide data.semantic_metadata_path pointing to class_descriptions_<dataset>.json. "
                "If using an explicit semantic_embedding_path, the .pt file must also contain "
                "class_semantics for validation."
            )

        records = []
        for idx, class_name in enumerate(self.class_names):
            if semantic_records is not None:
                if semantic_mapping is not None:
                    source = semantic_mapping.get(_semantic_key(class_name))
                    if source is None:
                        raise ValueError(
                            f"Semantic metadata is missing class '{class_name}' "
                            f"for {self.dataset_name}."
                        )
                else:
                    source = semantic_records[idx]
                record = _canonical_semantic_record(
                    source,
                    idx,
                    self.require_full_semantics,
                )
                record.setdefault("attribute_names", self.attribute_names)
                records.append(record)
                continue

            attr_c = self._format_attributes(attribute_raw[idx])
            record = {
                "name_c": class_name,
                "attr_c": attr_c,
                "name": class_name,
                "attributes": attr_c,
                "attr": attr_c,
                "attribute_names": self.attribute_names,
            }
            records.append(record)
        return records

    def get_class_attributes(self, class_indices=None):
        """Return attribute matrix for given class indices.

        Args:
            class_indices: array of original class indices. If None, return all.

        Returns:
            torch.Tensor of shape (len(class_indices), attr_dim)
        """
        if class_indices is None:
            return torch.from_numpy(self.attribute)
        return torch.from_numpy(self.attribute[class_indices])

    def get_class_semantics(self, class_indices=None):
        """Return t_c records with name_c, attr_c, hier_c, desc_c."""
        if class_indices is None:
            return list(self.class_semantics)
        return [self.class_semantics[int(i)] for i in class_indices]

    @property
    def attr_dim(self):
        return self.attribute.shape[1]

    @property
    def num_classes(self):
        return self.attribute.shape[0]

    @property
    def feature_dim(self):
        return self.features.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.mode == "image":
            raw_path = self.raw_image_paths[idx]
            img_path = self._resolved_image_paths.get(idx)
            if img_path is None:
                img_path = _resolve_image_path(raw_path, self.image_roots)
                if img_path is None:
                    roots = ", ".join(self.image_roots)
                    raise FileNotFoundError(
                        "Raw-image training requires real image files; no black-image fallback is used. "
                        f"Could not resolve: {raw_path}. Searched roots: {roots}"
                    )
                self._resolved_image_paths[idx] = img_path
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            if self.return_semantics:
                return img, label, self.class_semantics[int(label)]
            return img, label
        else:
            feature = self.features[idx]
            if self.return_semantics:
                return feature, label, self.class_semantics[int(label)]
            return feature, label


class AWA1Dataset(ZSLDataset):
    """Animals with Attributes 1 dataset."""

    def __init__(self, data_root, split="train", **kwargs):
        super().__init__(data_root, "AWA1", split=split, **kwargs)


class AWA2Dataset(ZSLDataset):
    """Animals with Attributes 2 dataset."""

    def __init__(self, data_root, split="train", **kwargs):
        super().__init__(data_root, "AWA2", split=split, **kwargs)


class CUBDataset(ZSLDataset):
    """Caltech-UCSD Birds 200-2011 dataset."""

    def __init__(self, data_root, split="train", **kwargs):
        super().__init__(data_root, "CUB", split=split, **kwargs)


class SUNDataset(ZSLDataset):
    """SUN Attribute dataset."""

    def __init__(self, data_root, split="train", **kwargs):
        super().__init__(data_root, "SUN", split=split, **kwargs)


class APYDataset(ZSLDataset):
    """Attribute Pascal and Yahoo dataset."""

    def __init__(self, data_root, split="train", **kwargs):
        super().__init__(data_root, "APY", split=split, **kwargs)


DATASET_REGISTRY = {
    "AWA1": AWA1Dataset,
    "AWA2": AWA2Dataset,
    "CUB": CUBDataset,
    "SUN": SUNDataset,
    "APY": APYDataset,
}


def build_zsl_dataset(data_root, dataset_name, split="train", **kwargs):
    """Build one of the supported xlsa17 ZSL datasets."""
    dataset_cls = DATASET_REGISTRY[normalize_dataset_name(dataset_name)]
    return dataset_cls(data_root, split=split, **kwargs)


def build_dataset(data_root, dataset_name, split="train", **kwargs):
    """Alias for build_zsl_dataset, kept for simple dataset factory imports."""
    return build_zsl_dataset(data_root, dataset_name, split=split, **kwargs)


def build_zsl_splits(
    data_root,
    dataset_name,
    semantic_embedding_path=None,
    semantic_metadata_path=None,
    require_full_semantics=False,
    **kwargs,
):
    """Build all standard splits for a ZSL dataset.

    Returns:
        dict with keys: 'trainval', 'test_seen', 'test_unseen'
        and shared metadata: 'attribute', 'seen_classes', 'unseen_classes'.
        The 'attribute' key is kept for backward compatibility; when
        semantic_embedding_path is provided, it contains cached semantic
        embeddings instead of the original dataset attribute matrix.
    """
    metadata_path = semantic_metadata_path or semantic_embedding_path
    semantic_records = load_semantic_records(metadata_path) if metadata_path else None

    split_kwargs = dict(kwargs)
    split_kwargs["semantic_records"] = semantic_records
    split_kwargs["require_full_semantics"] = require_full_semantics

    trainval = build_zsl_dataset(data_root, dataset_name, split="trainval", **split_kwargs)
    test_seen = build_zsl_dataset(data_root, dataset_name, split="test_seen", **split_kwargs)
    test_unseen = build_zsl_dataset(data_root, dataset_name, split="test_unseen", **split_kwargs)

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
        "semantic_embedding": semantic,
        "seen_classes": trainval.seen_classes,
        "unseen_classes": trainval.unseen_classes,
    }


def build_all_zsl_splits(data_root, semantic_embedding_paths=None, **kwargs):
    """Build standard splits for all supported xlsa17 datasets.

    Args:
        data_root: Root directory containing AWA1/AWA2/CUB/SUN/APY folders.
        semantic_embedding_paths: Optional dict mapping dataset name to a cached
            semantic embedding path.
    """
    semantic_embedding_paths = semantic_embedding_paths or {}
    return {
        name: build_zsl_splits(
            data_root,
            name,
            semantic_embedding_path=semantic_embedding_paths.get(name),
            **kwargs,
        )
        for name in SUPPORTED_DATASETS
    }
