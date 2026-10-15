"""CZSL benchmark loaders cho SPRN bench6 (MIT-States, C-GQA, VAW-CZSL).

Port phần data-loading từ OWCZSL/dataset.py (WACV 2025, Jayasekara et al.)
— giữ nguyên format split + protocol, bỏ các phần train-specific
(BERT tokenizer, precomputed .t7 features, affordance sampling).

Mọi ảnh + metadata đọc TRỰC TIẾP trên IBM (bench6_data/<name>/).

Classical framing (theo plan đã duyệt):
    Mỗi composition (state, object) = 1 lớp.
    __getitem__ trả (img_tensor, pair_idx) — đúng interface model(attr, features).
"""

import json
import os
from itertools import product
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SUPPORTED_CZSL = ("mit-states", "cgqa", "vaw-czsl", "ut-zap50k")

# CLIP test transform — đúng chuẩn OWCZSL test (Resize 224 BICUBIC + CenterCrop
# + CLIP normalize), phù hợp visual backbone v14 (CLIP ViT-B/16).
BICUBIC = transforms.InterpolationMode.BICUBIC


def clip_test_transform(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=BICUBIC),
            transforms.CenterCrop(image_size),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def clip_train_transform(image_size=224):
    resize_size = image_size + 32
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class ImageLoader:
    """Đọc ảnh từ <root>/images/<folder>/<file> — port từ OWCZSL (xử lý folder
    chứa dấu gạch dưới từ rename ' ' -> '_')."""

    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = f"{self.img_dir}/{img}"
        try:
            return Image.open(file).convert("RGB")
        except Exception:
            if "_" in file.split("/")[-2]:
                file = (
                    self.img_dir
                    + "/"
                    + file.split("/")[-2].replace("_", " ")
                    + "/"
                    + file.split("/")[-1]
                )
            return Image.open(file).convert("RGB")


class _CZSLTestDataset(Dataset):
    """Ảnh test + pair label. Bỏ mọi thứ train (text, kv, img1_a...)."""

    def __init__(self, data, loader, transform, pair2idx):
        self.data = data
        self.loader = loader
        self.transform = transform
        self.pair2idx = pair2idx

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)
        return img, self.pair2idx[(attr, obj)]

    def __len__(self):
        return len(self.data)


def _parse_pairs(pair_list, dataset_name):
    with open(pair_list, "r") as f:
        pairs = f.read().strip().split("\n")
        if dataset_name == "vaw-czsl":
            pairs = [t.split("+") for t in pairs]
        else:
            pairs = [t.split() for t in pairs]
        pairs = list(map(tuple, pairs))
    attrs, objs = zip(*pairs)
    return attrs, objs, pairs


def _split_name(dataset_name):
    """Tất cả 4 bộ dùng 'compositional-split-natural' (Purushwalkam split,
    verified trên data đã tải: mit-states/cgqa/vaw-czsl/ut-zap50k)."""
    return "compositional-split-natural"


def _parse_split(root_dir, dataset_name, split):
    """Port parse_split OWCZSL:117-141."""
    splitname = _split_name(dataset_name)
    split_dir = Path(root_dir) / splitname
    tr_attrs, tr_objs, tr_pairs = _parse_pairs(
        split_dir / "train_pairs.txt", dataset_name
    )
    vl_attrs, vl_objs, vl_pairs = _parse_pairs(
        split_dir / "val_pairs.txt", dataset_name
    )
    ts_attrs, ts_objs, ts_pairs = _parse_pairs(
        split_dir / "test_pairs.txt", dataset_name
    )
    all_attrs, all_objs = sorted(
        list(set(tr_attrs + vl_attrs + ts_attrs))
    ), sorted(list(set(tr_objs + vl_objs + ts_objs)))
    all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))
    return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs


def _load_metadata_t7(root_dir, splitname, pairs_set):
    """Port get_split_info OWCZSL:283-301 — đọc metadata_<split>.t7.

    metadata .t7 là torch.save list dict {image, attr, obj, set}.
    Trả (train_data, val_data, test_data) lọc như OWCZSL.
    """
    import torch as _torch

    path = Path(root_dir) / f"metadata_{splitname}.t7"
    data = _torch.load(str(path), weights_only=False)
    train_data, val_data, test_data = [], [], []
    for instance in data:
        image, attr, obj, settype = (
            instance["image"],
            instance["attr"],
            instance["obj"],
            instance["set"],
        )
        if attr == "NA" or (attr, obj) not in pairs_set or settype == "NA":
            continue
        if settype == "train":
            train_data.append([image, attr, obj])
        elif settype == "val":
            val_data.append([image, attr, obj])
        else:
            test_data.append([image, attr, obj])
    return train_data, val_data, test_data


def build_czsl_bench(name, data_root, image_size=224, universe="closed"):
    """Build CZSL bench cho SPRN.

    Args:
        name: "mit-states" | "cgqa" | "vaw-czsl" | "ut-zap50k"
        data_root: bench6_data/<name> (chứa images/, split files, metadata t7)
        image_size: 224 (CLIP).
        universe: "closed" → pairs = train∪val∪test (đúng vùng đánh giá
            closed-world, universe nhỏ cho phi/LLM); "full" → full product
            attr×obj (open-world universe OWCZSL, cần phi cho mọi pair).

    Returns dict:
        dataset_test: torch Dataset -> (img_tensor, pair_idx)
        attrs, objs, pairs, train/val/test_pairs, pair2idx, num_pairs
    """
    name = str(name).lower()
    if name not in SUPPORTED_CZSL:
        raise ValueError(f"Unsupported CZSL dataset '{name}'. Supported: {SUPPORTED_CZSL}")
    root_dir = str(Path(data_root).expanduser())

    attrs, objs, close_pairs, train_pairs, val_pairs, test_pairs = _parse_split(
        root_dir, name, "natural"
    )
    if universe == "full":
        pairs = sorted(list(product(attrs, objs)))  # open-world universe OWCZSL
    else:
        # closed universe: train ∪ val ∪ test (đúng vùng closed-world)
        pairs = sorted(
            set(train_pairs) | set(val_pairs) | set(test_pairs)
        )
    pair2idx = {pair: idx for idx, pair in enumerate(pairs)}

    # metadata .t7 theo split natural (Purushwalkam) cho cả 4 bộ.
    splitname = _split_name(name)
    train_data, val_data, test_data = _load_metadata_t7(
        root_dir, splitname, set(close_pairs)
    )

    loader = ImageLoader(os.path.join(root_dir, "images"))
    transform_test = clip_test_transform(image_size)
    transform_train = clip_train_transform(image_size)

    # Split test_data into seen (pairs in train_pairs) and unseen
    train_pair_set = set(train_pairs)
    test_seen_data = [d for d in test_data if (d[1], d[2]) in train_pair_set]
    test_unseen_data = [d for d in test_data if (d[1], d[2]) not in train_pair_set]

    # Fallback if test split has no seen samples (take from val/train)
    if not test_seen_data and val_data:
        test_seen_data = [d for d in val_data if (d[1], d[2]) in train_pair_set]
    if not test_seen_data and train_data:
        test_seen_data = train_data[:min(1000, len(train_data))]

    dataset_train = _CZSLTestDataset(train_data, loader, transform_train, pair2idx)
    dataset_test = _CZSLTestDataset(test_data, loader, transform_test, pair2idx)
    dataset_test_seen = _CZSLTestDataset(test_seen_data, loader, transform_test, pair2idx)
    dataset_test_unseen = _CZSLTestDataset(test_unseen_data, loader, transform_test, pair2idx)
    dataset_val = _CZSLTestDataset(val_data, loader, transform_test, pair2idx)

    seen_classes = sorted(list(set(pair2idx[p] for p in train_pairs if p in pair2idx)))
    unseen_classes = sorted(list(set(pair2idx[p] for p in pairs if p not in train_pair_set and p in pair2idx)))

    print(
        f"[{name}] #attrs={len(attrs)} #objs={len(objs)} "
        f"#train_pairs={len(train_pairs)} #val_pairs={len(val_pairs)} "
        f"#test_pairs={len(test_pairs)} #pairs={len(pairs)} (universe={universe}) "
        f"#train_images={len(train_data)} #test_images={len(test_data)} "
        f"#test_seen_imgs={len(test_seen_data)} #test_unseen_imgs={len(test_unseen_data)}"
    )
    return {
        "dataset_train": dataset_train,
        "dataset_test": dataset_test,
        "dataset_test_seen": dataset_test_seen,
        "dataset_test_unseen": dataset_test_unseen,
        "dataset_val": dataset_val,
        "attrs": list(attrs),
        "objs": list(objs),
        "pairs": pairs,
        "close_pairs": list(close_pairs),
        "train_pairs": list(train_pairs),
        "val_pairs": list(val_pairs),
        "test_pairs": list(test_pairs),
        "seen_classes": seen_classes,
        "unseen_classes": unseen_classes,
        "pair2idx": pair2idx,
        "num_pairs": len(pairs),
        "dataset_name": name,
    }
