#!/usr/bin/env python
"""Extract visual features for seen splits of XLSA17 ZSL datasets.

The tool reads XLSA17 ``res101.mat``/``att_splits.mat`` metadata, resolves raw
image paths from ``image_files``, and caches visual features to ``.pt`` files.
Only seen-class splits are exposed: train, trainval, val_seen, test_seen, and
all_seen.

Examples:
    python tools/extract_visual_features.py \
        --data_root ./data \
        --dataset AWA2 \
        --image_root /path/to/AWA2/JPEGImages \
        --output_root ./data_visual_features

    python tools/extract_visual_features.py \
        --data_root ./data \
        --dataset AWA2 \
        --image_root /path/to/AWA2 \
        --models resnet101 vit clip \
        --splits trainval test_seen \
        --feature_mode both \
        --layers 4,8,12
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))


SEEN_SPLITS = ("train", "trainval", "val_seen", "test_seen", "all_seen")
MODEL_CHOICES = ("resnet101", "vit", "clip")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIT_WEIGHT_ENUMS = {
    "vit_b_16": "ViT_B_16_Weights",
    "vit_b_32": "ViT_B_32_Weights",
    "vit_l_16": "ViT_L_16_Weights",
    "vit_l_32": "ViT_L_32_Weights",
}
CLIP_DESIGN_DETAILS = {
    "trainer": "",
    "vision_depth": 0,
    "vision_ctx": 0,
    "language_depth": 0,
    "language_ctx": 0,
}


class ImageRecordDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record["path"]).convert("RGB")
        return (
            self.transform(image),
            int(record["label"]),
            int(record["index"]),
            str(record["path"]),
            record["raw_image_file"],
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ResNet101/ViT/CLIP visual features for seen ZSL splits."
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root containing XLSA17 dataset folders.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset folder name, e.g. AWA1, AWA2, CUB, SUN, APY.")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Root containing raw images. If omitted, paths in res101.mat must exist.")
    parser.add_argument("--output_root", type=str, default="./data_visual_features",
                        help="Directory where .pt feature files are saved.")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES,
                        default=list(MODEL_CHOICES),
                        help="Visual encoders to run.")
    parser.add_argument("--splits", nargs="+", choices=SEEN_SPLITS,
                        default=["trainval"],
                        help="Seen split(s) to extract.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--vit_model", type=str, default="vit_b_16",
                        choices=sorted(VIT_WEIGHT_ENUMS.keys()),
                        help="Torchvision ViT factory used for --models vit.")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        help="CLIP model name or checkpoint path.")
    parser.add_argument("--feature_mode", choices=["pooled", "layers", "both"],
                        default="pooled",
                        help="Save final pooled features, layer features, or both.")
    parser.add_argument("--layers", type=str, default=None,
                        help=(
                            "Layer branch selection. For ResNet use names like "
                            "layer1,layer2,layer3,layer4. For ViT/CLIP use "
                            "1-based block numbers like 4,8,12. Defaults to "
                            "all ResNet stages or transformer thirds."
                        ))
    parser.add_argument("--layer_pool", choices=["pooled", "raw"], default="pooled",
                        help="For layer features, save pooled vectors or raw maps/tokens.")
    parser.add_argument("--normalize", action="store_true",
                        help="L2-normalize 2D feature tensors before saving.")
    parser.add_argument("--save_dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--skip_missing", action="store_true",
                        help="Skip missing images instead of raising an error.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files.")
    return parser.parse_args()


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


def parse_image_files(res101):
    if "image_files" not in res101:
        raise KeyError(
            "res101.mat does not contain image_files. Raw-image extraction needs "
            "XLSA17 image file metadata."
        )
    values = res101["image_files"].squeeze()
    return [matlab_to_string(value) for value in values]


def mat_index(att_splits, key):
    return att_splits[key].squeeze().astype(np.int64) - 1


def split_locations(att_splits, labels, split):
    trainval_loc = mat_index(att_splits, "trainval_loc")
    test_seen_loc = mat_index(att_splits, "test_seen_loc")
    seen_classes = set(np.unique(labels[trainval_loc]).tolist())

    if split == "train":
        loc = mat_index(att_splits, "train_loc") if "train_loc" in att_splits else trainval_loc
    elif split == "trainval":
        loc = trainval_loc
    elif split == "val_seen":
        loc = mat_index(att_splits, "val_loc") if "val_loc" in att_splits else test_seen_loc
    elif split == "test_seen":
        loc = test_seen_loc
    elif split == "all_seen":
        loc = np.unique(np.concatenate([trainval_loc, test_seen_loc]))
    else:
        raise ValueError(f"Unsupported seen split: {split}")

    return np.asarray([idx for idx in loc if int(labels[idx]) in seen_classes], dtype=np.int64)


def resolve_image_path(raw_path, roots):
    raw = str(raw_path).replace("\\", "/")
    direct = Path(raw_path)
    if direct.exists():
        return direct

    parts = [part for part in raw.split("/") if part and part != "."]
    candidates = []
    for root in roots:
        if root is None:
            continue
        root = Path(root)
        candidates.append(root / raw)
        if parts:
            for start in range(len(parts)):
                candidates.append(root.joinpath(*parts[start:]))
            candidates.append(root / parts[-1])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_seen_records(args, split):
    dataset_name = args.dataset.upper()
    data_root = Path(args.data_root)
    dataset_dir = data_root / dataset_name
    res_path = dataset_dir / "res101.mat"
    att_path = dataset_dir / "att_splits.mat"

    if not res_path.exists():
        raise FileNotFoundError(f"res101.mat not found: {res_path}")
    if not att_path.exists():
        raise FileNotFoundError(f"att_splits.mat not found: {att_path}")

    res101 = sio.loadmat(res_path)
    att_splits = sio.loadmat(att_path)
    labels = res101["labels"].squeeze().astype(np.int64) - 1
    image_files = parse_image_files(res101)
    if len(image_files) != len(labels):
        raise ValueError(
            f"image_files length ({len(image_files)}) does not match labels length ({len(labels)})."
        )

    loc = split_locations(att_splits, labels, split)
    roots = [args.image_root, dataset_dir, data_root]
    records = []
    missing = []

    for idx in loc:
        raw_image_file = image_files[int(idx)]
        path = resolve_image_path(raw_image_file, roots)
        if path is None:
            missing.append(raw_image_file)
            continue
        records.append({
            "index": int(idx),
            "label": int(labels[int(idx)]),
            "path": path,
            "raw_image_file": raw_image_file,
        })

    if missing and not args.skip_missing:
        examples = "\n  ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} images from split '{split}' could not be resolved. "
            f"Pass --image_root pointing to the raw image directory.\n  {examples}"
        )
    if not records:
        raise RuntimeError(f"No images available for split '{split}'.")

    return records


def imagenet_transform(image_size):
    resize_size = int(round(image_size / 0.875))
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_torchvision_weights(factory_name, enum_name, pretrained):
    factory = getattr(models, factory_name)
    if not pretrained:
        try:
            return factory(weights=None)
        except TypeError:
            return factory(pretrained=False)

    enum = getattr(models, enum_name, None)
    if enum is not None:
        return factory(weights=enum.DEFAULT)
    return factory(pretrained=True)


def build_resnet101(args, device):
    model = load_torchvision_weights("resnet101", "ResNet101_Weights", args.pretrained)
    model.to(device).eval()
    return {"kind": "resnet101", "model": model, "transform": imagenet_transform(args.image_size)}


def build_vit(args, device):
    enum_name = VIT_WEIGHT_ENUMS[args.vit_model]
    model = load_torchvision_weights(args.vit_model, enum_name, args.pretrained)
    model.to(device).eval()
    return {"kind": "vit", "model": model, "transform": imagenet_transform(args.image_size)}


def build_clip(args, device):
    if not args.pretrained:
        raise ValueError("CLIP extraction requires a checkpoint; remove --no_pretrained for --models clip.")

    model, preprocess = load_clip_for_visual(args.clip_model, device)
    model.eval()
    return {"kind": "clip", "model": model, "transform": preprocess}


def load_clip_for_visual(name, device):
    from sematics.clip.clip import _MODELS, _download, _transform
    from sematics.clip.model import build_model

    if name in _MODELS:
        model_path = _download(_MODELS[name], os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"CLIP model {name} not found. Available: {sorted(_MODELS)}")

    with open(model_path, "rb") as opened_file:
        try:
            jit_model = torch.jit.load(opened_file, map_location="cpu").eval()
            state_dict = jit_model.state_dict()
        except RuntimeError:
            state_dict = torch.load(opened_file, map_location="cpu")

    try:
        model = build_model(state_dict, design_details=CLIP_DESIGN_DETAILS).to(device)
    except TypeError:
        model = build_model(state_dict).to(device)

    if device.type == "cpu":
        model.float()
    return model, _transform(model.visual.input_resolution)


def build_extractor(model_name, args, device):
    if model_name == "resnet101":
        return build_resnet101(args, device)
    if model_name == "vit":
        return build_vit(args, device)
    if model_name == "clip":
        return build_clip(args, device)
    raise ValueError(f"Unknown model: {model_name}")


def parse_resnet_layers(spec):
    valid = ("layer1", "layer2", "layer3", "layer4")
    if not spec:
        return list(valid)
    layers = [item.strip() for item in spec.split(",") if item.strip()]
    invalid = [layer for layer in layers if layer not in valid]
    if invalid:
        raise ValueError(f"Invalid ResNet layer(s): {invalid}. Valid: {valid}")
    return layers


def parse_transformer_layers(spec, depth):
    if not spec:
        return sorted({max(1, depth // 3), max(1, (2 * depth) // 3), depth})

    layers = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        number = int(item)
        if number < 1 or number > depth:
            raise ValueError(f"Transformer layer {number} is outside 1..{depth}.")
        layers.append(number)
    return sorted(set(layers))


def pool_spatial(x, layer_pool):
    if layer_pool == "raw":
        return x
    return F.adaptive_avg_pool2d(x, output_size=1).flatten(1)


def pool_tokens(x, layer_pool):
    if layer_pool == "raw":
        return x
    return x[:, 0]


def extract_resnet101(model, images, args):
    want_layers = args.feature_mode in {"layers", "both"}
    selected = set(parse_resnet_layers(args.layers)) if want_layers else set()
    layers = {}

    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    if "layer1" in selected:
        layers["layer1"] = pool_spatial(x, args.layer_pool)
    x = model.layer2(x)
    if "layer2" in selected:
        layers["layer2"] = pool_spatial(x, args.layer_pool)
    x = model.layer3(x)
    if "layer3" in selected:
        layers["layer3"] = pool_spatial(x, args.layer_pool)
    x = model.layer4(x)
    if "layer4" in selected:
        layers["layer4"] = pool_spatial(x, args.layer_pool)

    final = None
    if args.feature_mode in {"pooled", "both"}:
        final = model.avgpool(x).flatten(1)
    return final, layers


def extract_torchvision_vit(model, images, args):
    layers_seq = model.encoder.layers
    depth = len(layers_seq)
    want_layers = args.feature_mode in {"layers", "both"}
    selected = set(parse_transformer_layers(args.layers, depth)) if want_layers else set()
    layers = {}

    x = model._process_input(images)
    batch_class_token = model.class_token.expand(x.shape[0], -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x = x + model.encoder.pos_embedding
    x = model.encoder.dropout(x)

    for idx, block in enumerate(layers_seq, start=1):
        x = block(x)
        if idx in selected:
            layers[f"block_{idx}"] = pool_tokens(x, args.layer_pool)

    x = model.encoder.ln(x)
    final = x[:, 0] if args.feature_mode in {"pooled", "both"} else None
    return final, layers


def extract_clip_resnet(visual, images, args):
    want_layers = args.feature_mode in {"layers", "both"}
    selected = set(parse_resnet_layers(args.layers)) if want_layers else set()
    layers = {}

    def stem(x):
        for conv, bn in [(visual.conv1, visual.bn1), (visual.conv2, visual.bn2), (visual.conv3, visual.bn3)]:
            x = visual.relu(bn(conv(x)))
        return visual.avgpool(x)

    x = images.type(visual.conv1.weight.dtype)
    x = stem(x)
    x = visual.layer1(x)
    if "layer1" in selected:
        layers["layer1"] = pool_spatial(x, args.layer_pool)
    x = visual.layer2(x)
    if "layer2" in selected:
        layers["layer2"] = pool_spatial(x, args.layer_pool)
    x = visual.layer3(x)
    if "layer3" in selected:
        layers["layer3"] = pool_spatial(x, args.layer_pool)
    x = visual.layer4(x)
    if "layer4" in selected:
        layers["layer4"] = pool_spatial(x, args.layer_pool)

    final = visual.attnpool(x) if args.feature_mode in {"pooled", "both"} else None
    return final, layers


def extract_clip_vit(visual, images, args):
    blocks = visual.transformer.resblocks
    depth = len(blocks)
    want_layers = args.feature_mode in {"layers", "both"}
    selected = set(parse_transformer_layers(args.layers, depth)) if want_layers else set()
    layers = {}

    x = images.type(visual.conv1.weight.dtype)
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype)
    cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)

    if getattr(visual, "VPT_shallow", False):
        visual_ctx = visual.VPT.expand(x.shape[0], -1, -1).to(dtype=x.dtype, device=x.device)
        x = torch.cat([x, visual_ctx], dim=1)

    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)

    for idx, block in enumerate(blocks, start=1):
        out = block(x)
        x = out[0] if isinstance(out, tuple) else out
        if idx in selected:
            tokens = x.permute(1, 0, 2)
            if getattr(visual, "VPT_shallow", False):
                tokens = tokens[:, :-visual.n_ctx]
            layers[f"block_{idx}"] = pool_tokens(tokens, args.layer_pool)

    tokens = x.permute(1, 0, 2)
    if getattr(visual, "VPT_shallow", False):
        tokens = tokens[:, :-visual.n_ctx]
    final = None
    if args.feature_mode in {"pooled", "both"}:
        final = visual.ln_post(tokens[:, 0, :])
        if visual.proj is not None:
            final = final @ visual.proj
    return final, layers


def extract_clip(model, images, args):
    visual = model.visual
    if hasattr(visual, "transformer") and hasattr(visual, "class_embedding"):
        return extract_clip_vit(visual, images, args)
    if hasattr(visual, "layer1") and hasattr(visual, "attnpool"):
        return extract_clip_resnet(visual, images, args)

    encoded = model.encode_image(images)
    if isinstance(encoded, tuple):
        encoded = next(item for item in encoded if torch.is_tensor(item) and item.dim() == 2)
    return encoded, {}


def postprocess_feature(tensor, args):
    if tensor is None:
        return None
    tensor = tensor.detach().cpu()
    if args.normalize and tensor.dim() == 2:
        tensor = F.normalize(tensor.float(), p=2, dim=-1)
    if args.save_dtype == "float16":
        return tensor.half()
    return tensor.float()


def extract_batch(bundle, images, args):
    kind = bundle["kind"]
    model = bundle["model"]
    if kind == "resnet101":
        return extract_resnet101(model, images, args)
    if kind == "vit":
        return extract_torchvision_vit(model, images, args)
    if kind == "clip":
        return extract_clip(model, images, args)
    raise ValueError(f"Unknown extractor kind: {kind}")


def output_path(args, model_name, split):
    dataset = args.dataset.upper()
    suffix = args.feature_mode
    return Path(args.output_root) / dataset / model_name / f"{split}_{suffix}.pt"


def save_split_features(args, model_name, split, records, bundle, device):
    out_path = output_path(args, model_name, split)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {out_path}. Use --overwrite to replace it.")

    dataset = ImageRecordDataset(records, bundle["transform"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    feature_chunks = []
    layer_chunks = {}
    label_chunks = []
    index_chunks = []
    image_paths = []
    raw_image_files = []

    desc = f"{args.dataset.upper()} {split} {model_name}"
    with torch.inference_mode():
        for images, labels, indices, paths, raw_paths in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            final, layers = extract_batch(bundle, images, args)

            final = postprocess_feature(final, args)
            if final is not None:
                feature_chunks.append(final)

            for name, tensor in layers.items():
                layer_chunks.setdefault(name, []).append(postprocess_feature(tensor, args))

            label_chunks.append(labels.cpu().long())
            index_chunks.append(indices.cpu().long())
            image_paths.extend(paths)
            raw_image_files.extend(raw_paths)

    output = {
        "dataset": args.dataset.upper(),
        "split": split,
        "model": model_name,
        "feature_mode": args.feature_mode,
        "layer_pool": args.layer_pool if args.feature_mode in {"layers", "both"} else None,
        "labels": torch.cat(label_chunks, dim=0),
        "indices": torch.cat(index_chunks, dim=0),
        "image_paths": image_paths,
        "raw_image_files": raw_image_files,
        "pretrained": bool(args.pretrained),
        "normalized": bool(args.normalize),
        "save_dtype": args.save_dtype,
    }

    if model_name == "vit":
        output["model_name"] = args.vit_model
    elif model_name == "clip":
        output["model_name"] = args.clip_model
    else:
        output["model_name"] = "resnet101"

    if feature_chunks:
        output["features"] = torch.cat(feature_chunks, dim=0)
        output["feature_shape"] = tuple(output["features"].shape)
    if layer_chunks:
        output["layer_features"] = {
            name: torch.cat(chunks, dim=0) for name, chunks in layer_chunks.items()
        }
        output["layer_shapes"] = {
            name: tuple(tensor.shape) for name, tensor in output["layer_features"].items()
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print(f"Saved {len(records)} samples to {out_path}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    split_records = {split: load_seen_records(args, split) for split in args.splits}
    for model_name in args.models:
        bundle = build_extractor(model_name, args, device)
        for split, records in split_records.items():
            save_split_features(args, model_name, split, records, bundle, device)


if __name__ == "__main__":
    main()
