#!/usr/bin/env python
"""Build a small prepared-dataset image subset and search HSPRN loss weights.

The subset is represented as a YAML manifest of randomly sampled split indices.
The search then trains each lambda candidate on the same manifest across
AWA2, CUB, SUN, and APY, and writes the best shared loss block to YAML.
"""

import argparse
import copy
import itertools
import random
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_DATASETS = ("AWA2", "CUB", "SUN", "APY")
SPLITS = ("trainval", "test_seen", "test_unseen")


def normalize_dataset_name(dataset_name):
    name = str(dataset_name).strip().upper()
    if name not in DEFAULT_DATASETS:
        supported = ", ".join(DEFAULT_DATASETS)
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported datasets: {supported}")
    return name


def parse_float_list(value):
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return [float(item) for item in values]


def parse_ratios(value):
    ratios = parse_float_list(value)
    if len(ratios) != len(SPLITS):
        raise argparse.ArgumentTypeError(
            f"Expected {len(SPLITS)} ratios for {', '.join(SPLITS)}."
        )
    total = sum(ratios)
    if total <= 0:
        raise argparse.ArgumentTypeError("Split ratios must sum to a positive value.")
    return [ratio / total for ratio in ratios]


def distribute(total, buckets):
    base, remainder = divmod(int(total), int(buckets))
    return [base + (1 if i < remainder else 0) for i in range(buckets)]


def split_budget(dataset_budget, ratios):
    raw = [dataset_budget * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = int(dataset_budget) - sum(counts)
    order = sorted(
        range(len(raw)),
        key=lambda i: raw[i] - counts[i],
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1
    return dict(zip(SPLITS, counts))


def resolve_package_dir(prepared_root, dataset_name):
    root = Path(prepared_root).expanduser().resolve()
    if root.name.upper() == dataset_name:
        return root
    return root / dataset_name


def build_candidates(args):
    grid = list(itertools.product(
        args.lambda1_values,
        args.lambda2_values,
        args.lambda3_values,
    ))
    candidates = [
        {
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
            "lambda3": float(lambda3),
        }
        for lambda1, lambda2, lambda3 in grid
    ]
    if args.search == "random":
        rng = random.Random(args.seed)
        rng.shuffle(candidates)
        candidates = candidates[: min(args.trials, len(candidates))]
    for candidate in candidates:
        for key, value in (
            ("temperature", args.temperature),
            ("align_mode", args.align_mode),
            ("pres_target", args.pres_target),
            ("pres_mode", args.pres_mode),
        ):
            if value is not None:
                candidate[key] = value
    return candidates


def sample_indices(length, count, rng):
    count = min(int(count), int(length))
    if count <= 0:
        return []
    return sorted(rng.sample(range(length), count))


def validate_image_subset(dataset, indices, dataset_name, split):
    for index in indices:
        try:
            dataset[index]
        except Exception as exc:
            raise RuntimeError(
                f"{dataset_name}/{split} failed image-mode validation at index {index}: {exc}"
            ) from exc


def build_dataset_subsets(args, dataset_names):
    from torch.utils.data import Subset

    from datasets import build_prepared_zsl_splits

    rng = random.Random(args.seed)
    per_dataset = distribute(args.total_size, len(dataset_names))
    ratios = parse_ratios(args.split_ratios)
    bundles = {}
    manifest = {
        "seed": int(args.seed),
        "total_requested": int(args.total_size),
        "source": "prepared",
        "prepared_root": str(Path(args.prepared_root).expanduser().resolve()),
        "mode": "image",
        "required_semantic_fields": ["name_c", "attr_c", "hier_c", "desc_c"],
        "datasets": {},
    }

    for dataset_name, budget in zip(dataset_names, per_dataset):
        package_dir = resolve_package_dir(args.prepared_root, dataset_name)
        metadata_path = package_dir / "metadata" / "class_semantics.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing prepared class semantics for {dataset_name}: {metadata_path}. "
                "The lambda-search subset requires name_c/attr_c/hier_c/desc_c."
            )

        split_counts = split_budget(budget, ratios)
        splits = build_prepared_zsl_splits(
            args.prepared_root,
            dataset_name,
            require_full_semantics=True,
            image_size=args.image_size,
            image_preprocess=args.image_preprocess,
        )

        selected = {}
        subset_manifest = {}
        for split in SPLITS:
            dataset = splits[split]
            indices = sample_indices(len(dataset), split_counts[split], rng)
            validate_image_subset(dataset, indices, dataset_name, split)
            selected[split] = Subset(dataset, indices)
            subset_manifest[split] = {
                "requested": int(split_counts[split]),
                "actual": int(len(indices)),
                "indices": [int(index) for index in indices],
            }

        bundles[dataset_name] = {
            "splits": splits,
            "subsets": selected,
        }
        manifest["datasets"][dataset_name] = {
            "requested": int(budget),
            "actual": int(sum(item["actual"] for item in subset_manifest.values())),
            "package_dir": str(package_dir),
            "semantic_metadata_path": str(metadata_path),
            "splits": subset_manifest,
        }

    manifest["total_actual"] = int(
        sum(info["actual"] for info in manifest["datasets"].values())
    )
    return bundles, manifest


def ensure_loss_schema(cfg, candidate):
    loss_cfg = cfg.model.loss
    for legacy_name in ("lambda4", "lambda_sem"):
        if hasattr(loss_cfg, legacy_name):
            delattr(loss_cfg, legacy_name)
    loss_cfg.lambda1 = float(candidate["lambda1"])
    loss_cfg.lambda2 = float(candidate["lambda2"])
    loss_cfg.lambda3 = float(candidate["lambda3"])
    for key in ("temperature", "align_mode", "pres_target", "pres_mode"):
        if candidate.get(key) is not None:
            setattr(loss_cfg, key, candidate[key])
    loss_cfg.return_full_output = True
    loss_cfg.use_full = True


def make_cfg(args, dataset_name, candidate, run_name):
    from tools.utils import load_config

    cfg = load_config(args.template_config, default_path=args.default_config)
    cfg.data.dataset = dataset_name
    cfg.data.source = "prepared"
    cfg.data.prepared_root = str(Path(args.prepared_root).expanduser().resolve())
    cfg.data.mode = "image"
    cfg.data.semantic_source = "class_desc"
    cfg.data.semantic_embedding_path = None
    cfg.data.semantic_metadata_path = None
    cfg.data.image_size = int(args.image_size)
    cfg.data.image_preprocess = args.image_preprocess
    cfg.data.subset_size = None
    cfg.data.eval_subset_size = None
    cfg.data.batch_size = int(args.batch_size)
    cfg.data.num_workers = int(args.num_workers)
    cfg.data.drop_last = False

    cfg.training.epochs = int(args.epochs)
    cfg.training.eval_freq = int(args.eval_freq)
    cfg.training.seed = int(args.seed)
    cfg.training.scheduler = args.scheduler
    cfg.training.save_dir = str(Path(args.work_dir) / "checkpoints" / run_name)
    cfg.training.log_dir = str(Path(args.work_dir) / "logs" / run_name)

    if args.semantic_backbone is not None:
        cfg.model.semantic.backbone_type = args.semantic_backbone
    if args.semantic_pretrained_weights is not None:
        cfg.model.semantic.pretrained_weights = args.semantic_pretrained_weights

    ensure_loss_schema(cfg, candidate)
    return cfg


def metric_value(stats, metric):
    if metric == "mean":
        return (float(stats["zsl_T1"]) + float(stats["cal_H"])) / 2.0
    return float(stats[metric])


def run_dataset_candidate(args, dataset_name, bundle, candidate, candidate_idx):
    import torch

    from losses import HSPRNLoss
    from model.hsprn_model import HSPRN
    from train import configure_semantic_input
    from trainer import Trainer
    from tools.utils import set_seed

    run_name = (
        f"candidate_{candidate_idx:03d}_{dataset_name}_"
        f"l1_{candidate['lambda1']}_l2_{candidate['lambda2']}_l3_{candidate['lambda3']}"
    ).replace(".", "p")
    cfg = make_cfg(
        args,
        dataset_name,
        candidate,
        run_name,
    )
    # Keep initialization and DataLoader order identical across candidates.
    set_seed(args.seed)

    trainval = bundle["subsets"]["trainval"]
    test_seen = bundle["subsets"]["test_seen"]
    test_unseen = bundle["subsets"]["test_unseen"]
    attribute = bundle["splits"]["attribute"]
    seen_classes = bundle["splits"]["seen_classes"]
    unseen_classes = bundle["splits"]["unseen_classes"]

    configure_semantic_input(cfg, attribute, "class_desc")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = HSPRN(cfg)
    criterion = HSPRNLoss(cfg)
    trainer = Trainer(
        cfg=cfg,
        model=model,
        criterion=criterion,
        train_dataset=trainval,
        test_seen_dataset=test_seen,
        test_unseen_dataset=test_unseen,
        attribute=attribute,
        seen_classes=seen_classes,
        unseen_classes=unseen_classes,
        device=device,
    )
    best_h, best_epoch = trainer.train()
    stats = trainer.last_eval_stats
    if stats is None:
        stats = trainer.evaluate_calibrated()
    stats = {key: float(value) for key, value in stats.items()}
    stats["best_H"] = float(best_h)
    stats["best_epoch"] = int(best_epoch)
    stats["score"] = metric_value(stats, args.metric)
    return stats


def run_search(args, dataset_names, bundles, candidates):
    results = []
    best = None

    for candidate_idx, candidate in enumerate(candidates, start=1):
        print(f"Candidate {candidate_idx}/{len(candidates)}: {candidate}")
        dataset_results = {}
        scores = []
        for dataset_name in dataset_names:
            print(f"  Dataset: {dataset_name}")
            stats = run_dataset_candidate(
                args,
                dataset_name,
                bundles[dataset_name],
                candidate,
                candidate_idx,
            )
            dataset_results[dataset_name] = stats
            scores.append(stats["score"])

        aggregate_score = float(sum(scores) / len(scores)) if scores else 0.0
        record = {
            "candidate_index": int(candidate_idx),
            "loss": copy.deepcopy(candidate),
            "score": aggregate_score,
            "datasets": dataset_results,
        }
        results.append(record)
        if best is None or record["score"] > best["score"]:
            best = record

    return best, results


def write_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced image subset from four prepared datasets and search "
            "lambda1/lambda2/lambda3."
        )
    )
    parser.add_argument("--prepared-root", default="./prepared_datasets")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--template-config", default="config/awa2_image_resnet50_train.yaml")
    parser.add_argument("--default-config", default="config/default.yaml")
    parser.add_argument("--subset-file", default="lambda_search_subset.yaml")
    parser.add_argument("--output", default="lambda_search_result.yaml")
    parser.add_argument("--work-dir", default="./lambda_search_runs")
    parser.add_argument("--total-size", type=int, default=200)
    parser.add_argument("--split-ratios", default="0.6,0.2,0.2")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-preprocess", default="imagenet", choices=["imagenet", "clip"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=1)
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine", "step"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--semantic-backbone",
        default=None,
        choices=["clip", "tinyclip", "sbert", "bert"],
        help="Optional text-backbone override for every search run.",
    )
    parser.add_argument(
        "--semantic-pretrained-weights",
        default=None,
        help="Optional local text-only model path used with --semantic-backbone.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search", default="random", choices=["grid", "random"])
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--lambda1-values", type=parse_float_list, default=[0.5, 0.7, 1.0])
    parser.add_argument("--lambda2-values", type=parse_float_list, default=[0.05, 0.1, 0.2])
    parser.add_argument("--lambda3-values", type=parse_float_list, default=[0.05, 0.1, 0.2])
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional loss temperature shared by every candidate.",
    )
    parser.add_argument(
        "--align-mode",
        default=None,
        choices=["cosine", "nce"],
        help="Optional L_align form shared by every candidate.",
    )
    parser.add_argument(
        "--pres-target",
        default=None,
        choices=["gt", "all"],
        help="Optional L_pres class target shared by every candidate.",
    )
    parser.add_argument(
        "--pres-mode",
        default=None,
        choices=["mse", "cosine"],
        help="Optional L_pres metric shared by every candidate.",
    )
    parser.add_argument(
        "--metric",
        default="mean",
        choices=["mean", "zsl_T1", "gzsl_H", "cal_H"],
        help="mean = average of ZSL T1 and calibrated GZSL H.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build/validate the subset manifest and candidate list.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_names = [
        normalize_dataset_name(name)
        for name in args.datasets.split(",")
        if name.strip()
    ]
    if not dataset_names:
        raise ValueError("At least one dataset must be selected.")

    bundles, manifest = build_dataset_subsets(args, dataset_names)
    candidates = build_candidates(args)
    write_yaml(args.subset_file, manifest)

    result = {
        "subset_file": args.subset_file,
        "data_source": "prepared",
        "prepared_root": str(Path(args.prepared_root).expanduser().resolve()),
        "datasets": dataset_names,
        "loss_mapping": {
            "lambda1": "L_cls",
            "lambda2": "L_align",
            "lambda3": "L_pres",
        },
        "selection_metric": args.metric,
        "selection_note": (
            "This search tunes on sampled official evaluation splits. Treat its "
            "scores as tuning diagnostics, not unbiased final test results."
        ),
        "search": args.search,
        "candidates": candidates,
        "status": "dry_run" if args.dry_run else "completed",
    }

    if args.dry_run:
        write_yaml(args.output, result)
        print(f"Wrote subset manifest to {args.subset_file}")
        print(f"Wrote dry-run search plan to {args.output}")
        return

    best, all_results = run_search(args, dataset_names, bundles, candidates)
    apply_to = []
    for dataset_name in dataset_names:
        pattern = f"{dataset_name.lower()}_image_*_train.yaml"
        apply_to.extend(
            str(path.relative_to(ROOT))
            for path in sorted((ROOT / "config").glob(pattern))
        )
    result.update({
        "best": copy.deepcopy(best),
        "all_results": all_results,
        "apply_to": apply_to,
        "model.loss": {
            "lambda1": best["loss"]["lambda1"],
            "lambda2": best["loss"]["lambda2"],
            "lambda3": best["loss"]["lambda3"],
            **{
                key: best["loss"][key]
                for key in ("temperature", "align_mode", "pres_target", "pres_mode")
                if best["loss"].get(key) is not None
            },
        },
    })
    write_yaml(args.output, result)
    print(f"Wrote subset manifest to {args.subset_file}")
    print(f"Wrote lambda search result to {args.output}")


if __name__ == "__main__":
    main()
