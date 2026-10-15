"""Multi-label benchmark loaders cho SPRN bench6 (MS-COCO, NUS-WIDE).

Port phần data-loading từ RCNn (ICCV 2025 'Language-Driven Multi-Label
Zero-Shot Learning with Semantic Granularity', Wang et al.) — giữ nguyên
protocol split:
  - MS-COCO: 48 seen / 17 unseen (cls_ids.pickle + instances_val2014_gzsi_48_17.json)
  - NUS-WIDE: seen = TagList1k \\ Concepts81 (925 tags), unseen = 81 human tags

Bỏ các phần train-specific của RCNn (CutImageplus, cfg node). Ảnh + anno
đọc TRỰC TIẾP trên IBM.

Classical framing: mỗi tag/lớp = 1 "class" cho semantic; model SPRN tính
score ảnh vs từng lớp. __getitem__ trả (img_tensor, multihot) — multihot
kích thước num_eval_classes (65 COCO / 81 NUS unseen).
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .czsl_dataset import clip_test_transform


# ---------------------------------------------------------------------------
# MS-COCO — port RCNn/datasets/coco.py
# ---------------------------------------------------------------------------

COCO_CLASSNAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class _CocoTestDataset(Dataset):
    """Ảnh test COCO (val2014, GZSL split) + multi-hot 65 lớp (48 seen + 17 unseen)."""

    def __init__(self, anno_path, img_path, transform):
        from pycocotools.coco import COCO

        cls_id = pickle.load(open(os.path.join(anno_path, "cls_ids.pickle"), "rb"))
        self.seen_cls_idx = sorted(list(cls_id["train"]))
        self.unseen_cls_idx = sorted(list(cls_id["test"]))
        self.eval_cls_idx = sorted(self.seen_cls_idx + self.unseen_cls_idx)  # 65
        self.eval_names = [COCO_CLASSNAMES[x] for x in self.eval_cls_idx]

        annFile = os.path.join(anno_path, "instances_val2014_gzsi_48_17.json")
        self.coco = COCO(annFile)
        self.img_path = img_path
        self.transform = transform
        self.ids = list(self.coco.imgToAnns.keys())

        self.cat2cat = dict()
        cats_keys = sorted([*self.coco.cats.keys()])
        for cat, cat2 in zip(cats_keys, self.eval_cls_idx):
            self.cat2cat[cat] = cat2

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        full = torch.zeros(len(COCO_CLASSNAMES), dtype=torch.float32)
        for obj in target:
            full[self.cat2cat[obj["category_id"]]] = 1
        # multihot trên eval_cls_idx (65)
        pos = [COCO_CLASSNAMES.index(COCO_CLASSNAMES[c]) for c in self.eval_cls_idx]
        multihot = full[self.eval_cls_idx]

        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.img_path, path)).convert("RGB")
        img = self.transform(img)
        return img, multihot

    def __len__(self):
        return len(self.ids)


class _CocoTrainDataset(Dataset):
    """Ảnh train COCO (train2014, 48 seen classes) + multi-hot 48 lớp."""

    def __init__(self, anno_path, img_path, transform):
        from pycocotools.coco import COCO

        cls_id = pickle.load(open(os.path.join(anno_path, "cls_ids.pickle"), "rb"))
        self.seen_cls_idx = sorted(list(cls_id["train"]))
        self.seen_names = [COCO_CLASSNAMES[x] for x in self.seen_cls_idx]

        annFile = os.path.join(anno_path, "instances_train2014_seen_48_17.json")
        self.coco = COCO(annFile)
        self.img_path = img_path
        self.transform = transform
        self.ids = list(self.coco.imgToAnns.keys())

        self.cat2idx = {}
        for idx, seen_idx in enumerate(self.seen_cls_idx):
            name = COCO_CLASSNAMES[seen_idx]
            for cat_id, cat_info in self.coco.cats.items():
                if cat_info["name"] == name:
                    self.cat2idx[cat_id] = idx
                    break

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        multihot = torch.zeros(len(self.seen_cls_idx), dtype=torch.float32)
        for obj in target:
            cat_id = obj["category_id"]
            if cat_id in self.cat2idx:
                multihot[self.cat2idx[cat_id]] = 1.0

        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.img_path, path)).convert("RGB")
        img = self.transform(img)
        return img, multihot

    def __len__(self):
        return len(self.ids)


def build_coco_train(data_root, image_size=224):
    from .czsl_dataset import clip_train_transform
    root = Path(data_root).expanduser()
    anno_path = str(root / "anno")
    img_path = str(root / "train2014")
    ds = _CocoTrainDataset(anno_path, img_path, clip_train_transform(image_size))
    return {
        "dataset_train": ds,
        "num_seen_classes": len(ds.seen_cls_idx),
        "seen_names": ds.seen_names,
        "dataset_name": "coco",
    }


def build_coco_bench(data_root, image_size=224):
    """data_root: bench6_data/coco (chứa val2014/ + anno/ từ RCNn)."""
    root = Path(data_root).expanduser()
    anno_path = str(root / "anno")
    img_path = str(root / "val2014")
    ds = _CocoTestDataset(anno_path, img_path, clip_test_transform(image_size))
    return {
        "dataset_test": ds,
        "class_names": ds.eval_names,
        "seen_cls_idx": ds.seen_cls_idx,
        "unseen_cls_idx": ds.unseen_cls_idx,
        "eval_cls_idx": ds.eval_cls_idx,
        "num_classes": len(ds.eval_cls_idx),
        "dataset_name": "coco",
    }


# ---------------------------------------------------------------------------
# NUS-WIDE — port RCNn/datasets/nuswide.py
# ---------------------------------------------------------------------------

def _nus_load_lines(path):
    with open(path, "r", errors="ignore") as f:
        return [l.rstrip().replace("\\", "/") for l in f.readlines()]


def _nus_labels_81(dir_81, tag81):
    label_tags = []
    for tag in tag81:
        with open(os.path.join(dir_81, "Labels_{}.txt".format(tag)), "r", errors="ignore") as f:
            lt = np.array(f.read().splitlines(), dtype=np.float32)[:, np.newaxis]
            label_tags.append(lt)
    return np.concatenate(label_tags, axis=1)


class _NuswideTestDataset(Dataset):
    """Ảnh test NUS-WIDE + multi-hot 81 unseen tags."""

    def __init__(self, anno_path, img_path, transform, data_set="test"):
        file_tag1k = os.path.join(anno_path, "TagList1k.txt")
        file_tag81 = os.path.join(anno_path, "Concepts81.txt")
        with open(file_tag1k, "r", errors="ignore") as f:
            tag1k = np.array(f.read().splitlines())
        with open(file_tag81, "r", errors="ignore") as f:
            tag81 = np.array(f.read().splitlines())
        # seen = tag1k \ tag81 (925), unseen = 81 human concepts
        self.unseen_names = list(tag81)
        self.num_unseen = len(tag81)

        id_filename = os.path.join(anno_path, "Imagelist.txt")
        data_partition = os.path.join(
            anno_path, "{}Imagelist.txt".format(data_set.title())
        )
        id_imgs = _nus_load_lines(id_filename)
        idxs_partition = _nus_load_lines(data_partition)

        dict_img_id = {}
        for idx, id_img in enumerate(id_imgs):
            dict_img_id[id_img.split("/")[-1]] = idx

        label81_imgs = _nus_labels_81(
            os.path.join(anno_path, "AllLabels"), tag81
        )

        self.img_names = []
        self.labels = []
        for img_id in idxs_partition:
            p = os.path.join(img_path, img_id)
            if not os.path.exists(p):
                flat_name = img_id.split("/")[-1]
                p_flat = os.path.join(img_path, flat_name)
                if os.path.exists(p_flat):
                    img_id = flat_name
                    p = p_flat
                else:
                    continue
            idx_dict = dict_img_id[img_id.split("/")[-1]]
            self.img_names.append(img_id)
            self.labels.append(label81_imgs[idx_dict])  # (81,) 0/1

        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = Image.open(os.path.join(self.img_path, img_name)).convert("RGB")
        img = self.transform(img)
        multihot = torch.from_numpy(self.labels[index]).float()
        return img, multihot

    def __len__(self):
        return len(self.img_names)


def build_nuswide_bench(data_root, image_size=224, transform=None):
    """data_root: bench6_data/nus-wide (chứa Flickr/ ảnh + anno NUS_WIDE).

    transform: cho PLAN backbone (native preprocess theo backbone);
    None -> clip_test_transform(image_size) như trước.
    """
    root = Path(data_root).expanduser()
    anno_path = str(root / "Groundtruth")
    img_path = str(root / "images")
    if not os.path.exists(img_path):
        img_path = str(root / "Flickr")
    tf = transform if transform is not None else clip_test_transform(image_size)
    ds = _NuswideTestDataset(anno_path, img_path, tf)
    return {
        "dataset_test": ds,
        "class_names": ds.unseen_names,
        "unseen_cls_idx": list(range(ds.num_unseen)),
        "num_classes": ds.num_unseen,
        "dataset_name": "nuswide",
    }


class _NuswideTrainDataset(Dataset):
    """Ảnh train NUS-WIDE (TrainImagelist, 161,789 images) + multi-hot 925 seen tags."""

    def __init__(self, data_root, transform):
        root = Path(data_root).expanduser()
        anno_path = root / "Groundtruth"
        img_path = root / "images"
        if not img_path.exists():
            img_path = root / "Flickr"

        mat_path = root / "Train_Tags925_seen.npy"
        if not mat_path.exists():
            raise FileNotFoundError(f"Missing {mat_path}")
        self.labels = np.load(mat_path, mmap_mode="r")

        train_imgs_file = anno_path / "TrainImagelist.txt"
        self.img_files = _nus_load_lines(train_imgs_file)
        self.img_path = str(img_path)
        self.transform = transform

    def __getitem__(self, index):
        img_id = self.img_files[index]
        p = os.path.join(self.img_path, img_id)
        if not os.path.exists(p):
            flat_name = img_id.split("/")[-1]
            p = os.path.join(self.img_path, flat_name)
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        multihot = torch.from_numpy(self.labels[index].copy()).float()
        return img, multihot

    def __len__(self):
        return len(self.img_files)


def build_nuswide_train(data_root, image_size=224):
    from .czsl_dataset import clip_train_transform
    ds = _NuswideTrainDataset(data_root, clip_train_transform(image_size))
    return {
        "dataset_train": ds,
        "num_seen_classes": 925,
        "dataset_name": "nuswide",
    }



def nus_gt_1006(test_ds, data_root):
    """Full 1006-column NUS-WIDE ground truth, aligned to ``test_ds`` row order.

    The loader only carries 81 unseen labels; the 925 seen columns come from
    ``Test_Tags1k.dat``. That file lives in ``NUS_WID_Tags/`` while the image
    list and ``Concepts81.txt`` live in ``Groundtruth/`` — the same split
    ``tools/eval_nuswide_gzsl_1006.py`` hardcoded. Row i of the tag matrix
    corresponds to line i of ``TestImagelist.txt``; rows are mapped by basename
    through ``test_ds.img_names`` (the loader drops images whose file is
    missing, so row order cannot be assumed).

    This is the single source of truth for NUS GT — the table that tool built
    (its ``build_gt_tables`` + ``loader_labels``), moved here so both entries
    share one implementation.
    """
    root = Path(data_root)
    tags_dir = root / "NUS_WID_Tags"
    gt_dir = root / "Groundtruth"
    tag1k = [l.strip() for l in open(tags_dir / "TagList1k.txt", errors="ignore") if l.strip()]
    tag81 = set(l.strip() for l in open(gt_dir / "Concepts81.txt", errors="ignore") if l.strip())
    seen_idx = [i for i, t in enumerate(tag1k) if t not in tag81]
    if len(seen_idx) != 925:
        raise ValueError(f"NUS GT: expected 925 seen tags, got {len(seen_idx)}.")

    tags = np.loadtxt(tags_dir / "Test_Tags1k.dat", dtype=np.float32)
    test_list = [l.strip() for l in open(gt_dir / "TestImagelist.txt", errors="ignore")
                 if l.strip()]
    row_of = {p.replace("\\", "/").split("/")[-1]: r for r, p in enumerate(test_list)}
    if len(row_of) != len(test_list):
        raise ValueError("NUS GT: duplicate basenames in TestImagelist.txt.")

    y_seen = tags[[row_of[n.replace("\\", "/").split("/")[-1]]
                   for n in test_ds.img_names]][:, seen_idx]
    y_unseen = np.stack([np.asarray(test_ds[i][1]) for i in range(len(test_ds))])
    y = np.concatenate([y_seen, y_unseen], axis=1)
    if y.shape != (len(test_ds), 1006):
        raise ValueError(f"NUS GT: built {y.shape}, expected {(len(test_ds), 1006)}.")
    return y
