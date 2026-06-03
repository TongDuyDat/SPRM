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

import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


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
    Proposed Split (PS).
    """

    def __init__(self, data_root, dataset_name, split="train"):
        """
        Args:
            data_root: Root directory containing dataset folders.
            dataset_name: One of 'AWA1', 'AWA2', 'CUB', 'SUN', 'APY'.
            split: 'train' | 'val_seen' | 'val_unseen' | 'test_seen' | 'test_unseen'.
        """
        super().__init__()
        self.dataset_name = normalize_dataset_name(dataset_name)
        self.split = split

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
        self.attribute = att_splits["att"].T.astype(np.float32)  # .mat stores as (attr_dim, C)
        self.attribute = self.attribute / (np.linalg.norm(self.attribute, axis=1, keepdims=True) + 1e-8)

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

        # Remap labels to contiguous seen-class indices for training
        self._build_label_maps()

    def _build_label_maps(self):
        """Build mappings between original labels and contiguous indices."""
        unique_labels = torch.unique(self.labels).numpy()
        self.label_to_idx = {int(l): i for i, l in enumerate(unique_labels)}
        self.idx_to_label = {i: int(l) for i, l in enumerate(unique_labels)}
        self.num_classes_in_split = len(unique_labels)

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
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label


class AWA1Dataset(ZSLDataset):
    """Animals with Attributes 1 dataset."""

    def __init__(self, data_root, split="train"):
        super().__init__(data_root, "AWA1", split=split)


class AWA2Dataset(ZSLDataset):
    """Animals with Attributes 2 dataset."""

    def __init__(self, data_root, split="train"):
        super().__init__(data_root, "AWA2", split=split)


class CUBDataset(ZSLDataset):
    """Caltech-UCSD Birds 200-2011 dataset."""

    def __init__(self, data_root, split="train"):
        super().__init__(data_root, "CUB", split=split)


class SUNDataset(ZSLDataset):
    """SUN Attribute dataset."""

    def __init__(self, data_root, split="train"):
        super().__init__(data_root, "SUN", split=split)


class APYDataset(ZSLDataset):
    """Attribute Pascal and Yahoo dataset."""

    def __init__(self, data_root, split="train"):
        super().__init__(data_root, "APY", split=split)


DATASET_REGISTRY = {
    "AWA1": AWA1Dataset,
    "AWA2": AWA2Dataset,
    "CUB": CUBDataset,
    "SUN": SUNDataset,
    "APY": APYDataset,
}


def build_zsl_dataset(data_root, dataset_name, split="train"):
    """Build one of the supported xlsa17 ZSL datasets."""
    dataset_cls = DATASET_REGISTRY[normalize_dataset_name(dataset_name)]
    return dataset_cls(data_root, split=split)


def build_dataset(data_root, dataset_name, split="train"):
    """Alias for build_zsl_dataset, kept for simple dataset factory imports."""
    return build_zsl_dataset(data_root, dataset_name, split=split)


def build_zsl_splits(data_root, dataset_name, semantic_embedding_path=None):
    """Build all standard splits for a ZSL dataset.

    Returns:
        dict with keys: 'trainval', 'test_seen', 'test_unseen'
        and shared metadata: 'attribute', 'seen_classes', 'unseen_classes'.
        The 'attribute' key is kept for backward compatibility; when
        semantic_embedding_path is provided, it contains cached semantic
        embeddings instead of the original dataset attribute matrix.
    """
    trainval = build_zsl_dataset(data_root, dataset_name, split="trainval")
    test_seen = build_zsl_dataset(data_root, dataset_name, split="test_seen")
    test_unseen = build_zsl_dataset(data_root, dataset_name, split="test_unseen")

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


def build_all_zsl_splits(data_root, semantic_embedding_paths=None):
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
        )
        for name in SUPPORTED_DATASETS
    }
