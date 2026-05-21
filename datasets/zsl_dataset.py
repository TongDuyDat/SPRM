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
        self.dataset_name = dataset_name.upper()
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
        trainval_loc = att_splits["trainval_loc"].squeeze() - 1
        test_seen_loc = att_splits["test_seen_loc"].squeeze() - 1
        test_unseen_loc = att_splits["test_unseen_loc"].squeeze() - 1

        # Train/val split from trainval: use 80/20 for validation
        train_loc = att_splits.get("train_loc", None)
        val_loc = att_splits.get("val_loc", None)

        if train_loc is not None:
            train_loc = train_loc.squeeze() - 1
        if val_loc is not None:
            val_loc = val_loc.squeeze() - 1

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


def build_zsl_splits(data_root, dataset_name):
    """Build all standard splits for a ZSL dataset.

    Returns:
        dict with keys: 'trainval', 'test_seen', 'test_unseen'
        and shared metadata: 'attribute', 'seen_classes', 'unseen_classes'
    """
    trainval = ZSLDataset(data_root, dataset_name, split="trainval")
    test_seen = ZSLDataset(data_root, dataset_name, split="test_seen")
    test_unseen = ZSLDataset(data_root, dataset_name, split="test_unseen")

    return {
        "trainval": trainval,
        "test_seen": test_seen,
        "test_unseen": test_unseen,
        "attribute": torch.from_numpy(trainval.attribute),
        "seen_classes": trainval.seen_classes,
        "unseen_classes": trainval.unseen_classes,
    }
