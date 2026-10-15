#!/usr/bin/env python
"""SPRN (LPR) evaluation entry point.

The protocol is fixed by the benchmark family (``data.type``):

    PreparedZSL      -> ZSLPoint        classic ZSL/GZSL Top-1, u/s/H, calibrated
    CZSLSplit        -> CZSLSweep       CZSL bias curve -> Best S/U/HM, AUC
    MultilabelSplit  -> MultilabelTopK  multi-label top-k P/R/F1 + mAP

Metric formulas live in tools/evaluator.py, tools/ml_evaluator.py,
tools/czsl_exact_metrics.py and tools/czsl_evaluator.py, wrapped by tools/metrics.py.

    python eval.py --dataset cub --checkpoint checkpoints/lpr/cub/CUB/final.pth
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from tools.metrics import METRICS
from tools.registry import DATASETS
from tools.utils import compose_config, set_seed

import datasets.builders  # noqa: F401  (registers the builders)
import tools.metrics  # noqa: F401  (registers the metrics)


METRIC_OF = {"PreparedZSL": "ZSLPoint", "CZSLSplit": "CZSLSweep",
             "MultilabelSplit": "MultilabelTopK"}


def parse_args():
    p = argparse.ArgumentParser(description="SPRN (LPR) evaluation")
    p.add_argument("--dataset", required=True, help="config/datasets/<dataset>.yaml (e.g. cub)")
    p.add_argument("--method", default="lpr")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None, help="Write metrics JSON here (parent dirs created).")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N test samples (smoke test).")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def _score_loader(model, loader, phi, class_indices, device):
    """Scores for one loader against ``class_indices`` rows of phi.

    Returns (scores [N, len(class_indices)], labels). The model consumes an attribute
    dict keyed by GLOBAL class id; score columns follow ``class_indices`` order.
    """
    model.eval()
    attr = {str(int(c)): phi[int(c)].to(device) for c in class_indices}
    chunks, labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, target = batch[0], batch[1]
            out = model(attr, imgs.to(device))
            chunks.append(out.scores.float().cpu())
            labels.append(target)
    return torch.cat(chunks, dim=0), torch.cat(labels, dim=0)


def _limited(ds, limit):
    return Subset(ds, range(min(limit, len(ds)))) if limit else ds


def _loader(ds, bs, nw, device):
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                      pin_memory=device == "cuda")


def eval_classic(args, cfg, built, model, phi, device):
    bs = int(args.batch_size or cfg.data.batch_size)
    nw = int(cfg.data.num_workers)
    seen_loader = _loader(_limited(built["test_seen"], args.limit), bs * 2, nw, device)
    unseen_loader = _loader(_limited(built["test_unseen"], args.limit), bs * 2, nw, device)

    seen_classes, unseen_classes = built["seen_classes"], built["unseen_classes"]
    num_total = int(phi.shape[0])
    all_classes = list(map(int, seen_classes)) + list(map(int, unseen_classes))

    # Score in the full class space with -inf outside the candidate set, which is what
    # ZSLEvaluator expects (and what makes seen/unseen comparable).
    def full_scores(loader, desc):
        sub, labels = _score_loader(model, loader, phi, all_classes, device)
        full = torch.full((sub.size(0), num_total), -float("inf"))
        for i, c in enumerate(all_classes):
            full[:, c] = sub[:, i]
        print(f"  {desc}: {tuple(full.shape)}")
        return full, labels

    scores_seen, labels_seen = full_scores(seen_loader, "scored seen")
    scores_unseen, labels_unseen = full_scores(unseen_loader, "scored unseen")
    metric = METRICS.build("ZSLPoint", seen_classes=seen_classes, unseen_classes=unseen_classes)
    return metric.compute(scores_seen, labels_seen, scores_unseen, labels_unseen)


def eval_czsl(args, cfg, built, model, phi, device):
    bench = built["bench"]
    bs = int(args.batch_size or cfg.data.batch_size)
    loader = _loader(_limited(bench["dataset_test"], args.limit), bs, int(cfg.data.num_workers), device)
    scores, labels = _score_loader(model, loader, phi, list(range(phi.shape[0])), device)
    print(f"  scored {tuple(scores.shape)}")
    metric = METRICS.build("CZSLSweep", bench=bench, universe="closed")
    return metric.compute(scores, labels)


def eval_multilabel(args, cfg, built, model, phi, device):
    import numpy as np

    test_ds = built["test_unseen"]
    bs = int(args.batch_size or cfg.data.batch_size)
    loader = _loader(_limited(test_ds, args.limit), bs, int(cfg.data.num_workers), device)
    scores, _ = _score_loader(model, loader, phi, list(range(phi.shape[0])), device)
    print(f"  scored {tuple(scores.shape)}")
    targets = np.asarray(test_ds.eval_targets, dtype=np.float32)[: scores.shape[0]]
    metric = METRICS.build("MultilabelTopK", unseen_pos=list(map(int, built["unseen_classes"])))
    return metric.compute(scores.numpy(), targets)


def main():
    args = parse_args()
    cfg, _ = compose_config(args.dataset, args.method)
    if args.seed is not None:
        cfg.training.seed = args.seed
    set_seed(cfg.training.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    data_type = cfg.data.type
    metric_type = METRIC_OF[data_type]
    print("=" * 70)
    print(f"SPRN eval | data.type={data_type} eval.type={metric_type}")
    print(f"dataset={args.dataset} ckpt={args.checkpoint}")
    print("=" * 70)

    # The builder also sets cfg.model.semantic.input_dim (classic), which the model needs.
    built = DATASETS.build(data_type, cfg)

    from model.sprn_model import SPRN

    model = SPRN(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    phi = built["attribute"].detach().float().to(device)

    fn = {"PreparedZSL": eval_classic, "CZSLSplit": eval_czsl, "MultilabelSplit": eval_multilabel}[data_type]
    out = fn(args, cfg, built, model, phi, device)

    # Coverage check: every declared cell must be present and non-None.
    missing = [k for k in getattr(METRICS.get(metric_type), "provides", ()) if out.get(k) is None]

    name = Path(args.checkpoint).parent.name
    printable = {k: v for k, v in out.items() if k != "curve"}
    print(f"\nRESULT {name}")
    print(json.dumps(printable, indent=2, default=str))
    if missing:
        raise SystemExit(f"COVERAGE_FAIL: {missing}")

    if args.output:
        dst = Path(args.output)
        dst.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(printable)
        payload["_dataset"] = args.dataset
        payload["_checkpoint"] = args.checkpoint
        payload["_checkpoint_epoch"] = ckpt.get("epoch")
        dst.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote {dst}")


if __name__ == "__main__":
    main()
