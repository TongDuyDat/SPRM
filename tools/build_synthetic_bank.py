#!/usr/bin/env python
"""v12 Phase S0 — build the synthetic-unseen backbone bank (offline, ~2 min).

One protocol-safe pass over TRAINVAL ONLY (frozen CLIP backbone, eval
transform) accumulates per-seen-class statistics of the backbone outputs:

  - cls  raw [512]        (bit-exact CLIP CLS, the MFA anchor input)
  - gap  raw [3,768]      (GAP of the 3 token levels; gap == mean(dim=1))
  - T_h  token map [197,768] (deepest level, the IPR spatial input)

Then the class-level statistics are TRANSFERRED to the unseen classes from
the frozen phi embeddings alone (Nadaraya-Watson kernel on phi, or ridge),
with the estimator chosen on a seen-class holdout (30/150 for CUB).

GATE (plan §3.5): cos_mean >= 0.80 AND nn_top1 >= 30% — on failure the script
exits non-zero WITHOUT saving, so no training budget is spent on a bad bank.

Usage (IBM):
    python tools/build_synthetic_bank.py \
        --config config/cub_clip_v11_e4_b16.yaml \
        --out ./pre-trained/cub_v12_synbank_b16.pt
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from tools.utils import load_config, set_seed

from datasets.prepared_dataset import build_prepared_zsl_dataset
from datasets.zsl_dataset import load_semantic_embeddings
from model.vision_model import VisionModel


def parse_args():
    p = argparse.ArgumentParser(
        description="Build the v12 synthetic-unseen backbone bank")
    p.add_argument("--config", type=str, required=True,
                   help="YAML config (dataset + vision backbone settings)")
    p.add_argument("--out", type=str, required=True, help="Output .pt path")
    p.add_argument("--holdout_frac", type=float, default=0.2,
                   help="Fraction of seen classes held out for estimator "
                        "selection (CUB: 0.2 -> 30/150)")
    p.add_argument("--nw_h", type=str, default="0.05,0.10,0.20",
                   help="Comma list of NW kernel bandwidths on cos(phi)")
    p.add_argument("--ridge_lambda", type=str, default="0.1,1,10",
                   help="Comma list of ridge lambdas")
    p.add_argument("--std_scale", type=float, default=1.0,
                   help="Scale applied to cls/gap std at save time "
                        "(Run-2 lever; 1.0 keeps the shrunk estimates)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase A — protocol-safe trainval extraction (frozen backbone, eval transform)
# ---------------------------------------------------------------------------

def extract_trainval_stats(cfg, device):
    dataset_name = cfg.data.dataset
    prepared_root = getattr(cfg.data, "prepared_root", "./prepared_datasets")

    # Protocol audit: trainval ONLY. Test splits are never constructed here.
    ds = build_prepared_zsl_dataset(
        prepared_root, dataset_name, split="trainval",
        image_size=int(cfg.data.image_size),
        image_preprocess=str(cfg.data.image_preprocess),
    )
    assert ds.split == "trainval"
    print(f"[Phase A] {dataset_name} trainval: {len(ds)} images, "
          f"{len(ds.seen_classes)} seen / {len(ds.unseen_classes)} unseen classes")

    # Force the eval transform exactly like trainer._build_null_condition.
    ds.transform = ds._default_transform("test")

    vision = VisionModel(cfg).to(device)
    vision.eval()

    loader = torch.utils.data.DataLoader(
        ds, batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers), shuffle=False,
    )

    seen_ids = [int(c) for c in ds.seen_classes]
    row_of = {cid: i for i, cid in enumerate(seen_ids)}
    n_seen = len(seen_ids)

    cls_sum = torch.zeros(n_seen, 512, device=device)
    cls_sqsum = torch.zeros(n_seen, 512, device=device)
    gap_sum = torch.zeros(n_seen, 3, 768, device=device)
    gap_sqsum = torch.zeros(n_seen, 3, 768, device=device)
    tok_sum = torch.zeros(n_seen, 197, 768, device=device)
    counts = torch.zeros(n_seen, device=device)

    all_cls = []      # per-image raw cls (Phase C holdout metrics)
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            out = vision(features)
            if not isinstance(out, dict):
                raise ValueError(
                    "build_synthetic_bank requires the anchored vision path "
                    "(vision.cls_anchor=true)."
                )
            cls_batch = out["cls"].float()               # [B,512]
            gaps = torch.stack(
                [t.float().mean(dim=1) for t in out["tokens"]], dim=1
            )                                            # [B,3,768]
            tokens_h = out["tokens"][-1].float()          # [B,197,768]

            all_cls.append(cls_batch.cpu())
            all_labels.append(labels.clone())

            for label in labels.unique().tolist():
                row = row_of.get(int(label))
                if row is None:
                    continue  # trainval contains only seen labels; safety
                mask = labels == label
                cls_sum[row] += cls_batch[mask].sum(dim=0)
                cls_sqsum[row] += (cls_batch[mask] ** 2).sum(dim=0)
                gap_sum[row] += gaps[mask].sum(dim=0)
                gap_sqsum[row] += (gaps[mask] ** 2).sum(dim=0)
                tok_sum[row] += tokens_h[mask].sum(dim=0)
                counts[row] += float(mask.sum())

    if int((counts == 0).sum()) > 0:
        raise RuntimeError("Some seen classes received no trainval images.")

    stats = {
        "seen_ids": seen_ids,
        "unseen_ids": [int(c) for c in ds.unseen_classes],
        "cls_mean": (cls_sum / counts.unsqueeze(1)).cpu(),
        "gap_mean": (gap_sum / counts.view(-1, 1, 1)).cpu(),
        "tok_mean_h": (tok_sum / counts.view(-1, 1, 1)).cpu(),
        "counts": counts.cpu(),
        "img_cls": torch.cat(all_cls, dim=0),   # [N,512] raw
        "img_labels": torch.cat(all_labels, dim=0),
    }

    # Raw diag variances (clamped against negative round-off).
    stats["cls_var"] = (
        cls_sqsum.cpu() / counts.unsqueeze(1).cpu() - stats["cls_mean"] ** 2
    ).clamp(min=0.0)
    stats["gap_var"] = (
        gap_sqsum.cpu() / counts.view(-1, 1, 1).cpu() - stats["gap_mean"] ** 2
    ).clamp(min=0.0)

    # Consistency check (plan §1.4): gap is linear, so the level-h class mean
    # must equal GAP of the class token mean.
    gap_from_tok = stats["tok_mean_h"].mean(dim=1)     # [n_seen,768]
    err = (gap_from_tok - stats["gap_mean"][:, 2, :]).abs().max().item()
    print(f"[Phase A] consistency |GAP(T_h) - mu_gap_h| max err = {err:.2e}")
    if err > 1e-2:
        raise RuntimeError("GAP/token-map consistency check failed.")
    return stats


# ---------------------------------------------------------------------------
# Phase B — shrunk diagonal variances
# ---------------------------------------------------------------------------

def shrink_variances(stats, min_count=20):
    """0.9 * sigma^2_s + 0.1 * mean(sigma^2); small classes -> global mean var."""
    counts = stats["counts"]
    small = counts < min_count
    n_small = int(small.sum())
    if n_small:
        print(f"[Phase B] WARNING: {n_small} seen classes have < {min_count} "
              f"images -> using the global mean variance for them.")

    out = {}
    for key in ("cls_var", "gap_var"):
        var = stats[key].clone()
        # Per-dimension mean variance over ALL seen classes (reliable estimate).
        mean_var = stats[key].mean(dim=0, keepdim=True)
        if n_small:
            var[small] = mean_var.expand(int(small.sum()), *var.shape[1:])
        var = 0.9 * var + 0.1 * mean_var
        out[key] = var
    return out


# ---------------------------------------------------------------------------
# Estimators (phi rows follow the SEEN/UNSEEN row order of the stats, not
# global class ids — callers pass phi_seen/phi_unseen already re-indexed)
# ---------------------------------------------------------------------------

def nw_weights(phi_u, phi_bank, h):
    """Nadaraya-Watson weights over `phi_bank` for query `phi_u` (cos kernel)."""
    sims = F.cosine_similarity(phi_u.unsqueeze(0), phi_bank, dim=-1)
    logits = sims / h
    logits = logits - logits.max()
    w = torch.exp(logits)
    return w / w.sum()


def nw_smooth(w, target):
    """Weighted mean over dim 0 of a [n_bank, ...] target."""
    view = w.view(-1, *([1] * (target.dim() - 1)))
    return (view * target).sum(dim=0)


def fit_ridge(phi_fit, target, lam, phi_mean, phi_std):
    """Ridge map mu = standardize(phi) @ W (+ bias column); target 2D."""
    x = (phi_fit - phi_mean) / phi_std
    x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    eye = torch.eye(x.shape[1])
    eye[-1, -1] = 0.0  # do not penalize the bias
    return torch.linalg.solve(x.T @ x + lam * eye, x.T @ target)


def apply_ridge(w, phi, phi_mean, phi_std):
    x = (phi - phi_mean) / phi_std
    x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    return x @ w


# ---------------------------------------------------------------------------
# Phase C — estimator selection on the seen-class holdout
# ---------------------------------------------------------------------------

def eval_holdout(kind, params, phi_seen, stats, hold_rows, fit_rows):
    """Predict mu_cls for the holdout classes; return (cos_mean, nn_top1).

    Metrics on the REAL trainval images of the holdout classes:
      cos_mean: cos(predicted mu vs real mu), averaged over holdout classes.
      nn_top1:  nearest-centroid over the bank {real mu of the pseudo-train
                classes + predicted mu of the holdout classes}.
    """
    n_fit = len(fit_rows)
    mu_real = stats["cls_mean"]
    pred = torch.zeros(len(hold_rows), mu_real.shape[1])
    if kind == "oracle":
        # Upper bound: the REAL holdout means. Not an estimator — it says how
        # much of nn_top1 is lost to the 150-way task itself rather than to
        # the transfer, so the 30% gate can be read in context.
        pred = mu_real[hold_rows].clone()
    elif kind == "nw":
        phi_fit = phi_seen[fit_rows]
        for i, r in enumerate(hold_rows):
            pred[i] = nw_smooth(nw_weights(phi_seen[r], phi_fit, params["h"]),
                               mu_real[fit_rows])
    else:  # ridge — standardize on all 150 seen, fit on the pseudo-train
        lam = params["lambda"]
        phi_mean = phi_seen.mean(dim=0)
        phi_std = phi_seen.std(dim=0).clamp(min=1e-6)
        w = fit_ridge(phi_seen[fit_rows], mu_real[fit_rows], lam,
                      phi_mean, phi_std)
        pred = apply_ridge(w, phi_seen[hold_rows], phi_mean, phi_std)

    real = mu_real[hold_rows]
    cos_mean = F.cosine_similarity(pred, real, dim=-1).mean().item()

    # Nearest-centroid proxy on real holdout-class trainval images.
    hold_pos = {stats["seen_ids"][r]: j for j, r in enumerate(hold_rows)}
    bank = torch.cat([mu_real[fit_rows], pred], dim=0)
    bank_n = F.normalize(bank, dim=-1)
    img_mask = torch.tensor(
        [int(l) in hold_pos for l in stats["img_labels"].tolist()]
    )
    imgs = F.normalize(stats["img_cls"][img_mask], dim=-1)
    img_labels = stats["img_labels"][img_mask]
    top1 = (imgs @ bank_n.T).argmax(dim=-1)
    target_col = torch.tensor(
        [n_fit + hold_pos[int(l)] for l in img_labels.tolist()]
    )
    nn_top1 = float((top1 == target_col).float().mean().item())
    return cos_mean, nn_top1


def select_estimator(args, phi_seen, stats):
    n_seen = len(stats["seen_ids"])
    n_hold = max(1, int(round(args.holdout_frac * n_seen)))
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_seen)
    hold_rows = sorted(int(r) for r in perm[:n_hold])
    hold_set = set(hold_rows)
    fit_rows = [r for r in range(n_seen) if r not in hold_set]
    print(f"[Phase C] holdout: {len(hold_rows)}/{n_seen} seen classes "
          f"(global ids {[stats['seen_ids'][r] for r in hold_rows]})")

    grid = []
    orc_cos, orc_nn = eval_holdout("oracle", {}, phi_seen, stats,
                                   hold_rows, fit_rows)
    print(f"  ORACLE (real holdout means)  cos_mean={orc_cos:.4f}  "
          f"nn_top1={orc_nn:.4f}  <- ceiling of this 150-way proxy")
    for h in [float(v) for v in args.nw_h.split(",") if v.strip()]:
        cos_m, nn = eval_holdout("nw", {"h": h}, phi_seen, stats,
                                 hold_rows, fit_rows)
        grid.append({"kind": "nw", "params": {"h": h},
                     "cos_mean": cos_m, "nn_top1": nn})
        print(f"  NW    h={h:<5}  cos_mean={cos_m:.4f}  nn_top1={nn:.4f}")
    for lam in [float(v) for v in args.ridge_lambda.split(",") if v.strip()]:
        cos_m, nn = eval_holdout("ridge", {"lambda": lam}, phi_seen, stats,
                                 hold_rows, fit_rows)
        grid.append({"kind": "ridge", "params": {"lambda": lam},
                     "cos_mean": cos_m, "nn_top1": nn})
        print(f"  RIDGE lam={lam:<5} cos_mean={cos_m:.4f}  nn_top1={nn:.4f}")

    # Winner: max nn_top1, tie-break cos_mean.
    best = max(grid, key=lambda g: (g["nn_top1"], g["cos_mean"]))
    print(f"[Phase C] winner: {best['kind']} {best['params']} "
          f"cos_mean={best['cos_mean']:.4f} nn_top1={best['nn_top1']:.4f}")

    gate = best["cos_mean"] >= 0.80 and best["nn_top1"] >= 0.30
    print(f"[Phase C] GATE cos_mean>=0.80 AND nn_top1>=0.30: "
          f"{'PASS' if gate else 'FAIL'}")
    return best, grid, hold_rows, gate


# ---------------------------------------------------------------------------
# Phase D — refit the winner on all seen classes, transfer to unseen
# ---------------------------------------------------------------------------

def transfer_mu(kind, params, phi_seen, phi_unseen, target):
    """Transfer a per-seen-class statistic [n_seen, ...] to the unseen."""
    if kind == "nw":
        out = []
        for u in range(phi_unseen.shape[0]):
            w = nw_weights(phi_unseen[u], phi_seen, params["h"])
            out.append(nw_smooth(w, target))
        return torch.stack(out, dim=0)
    # Ridge on the flattened statistic (targets beyond cls are >2D).
    flat = target.reshape(target.shape[0], -1)
    phi_mean = phi_seen.mean(dim=0)
    phi_std = phi_seen.std(dim=0).clamp(min=1e-6)
    w = fit_ridge(phi_seen, flat, params["lambda"], phi_mean, phi_std)
    pred = apply_ridge(w, phi_unseen, phi_mean, phi_std)
    return pred.view(phi_unseen.shape[0], *target.shape[1:])


def transfer_var(kind, params, phi_seen, phi_unseen, var_seen):
    """Transfer the shrunk diag variances to the unseen classes.

    A ridge mean-map does not define a variance map, so variance always goes
    through NW smoothing (h from the best NW entry of the grid); the choice
    is recorded in the provenance.
    """
    flat = var_seen.reshape(var_seen.shape[0], -1)
    out = []
    for u in range(phi_unseen.shape[0]):
        w = nw_weights(phi_unseen[u], phi_seen, params["h"])
        out.append(nw_smooth(w, flat))
    return torch.stack(out, dim=0).view(phi_unseen.shape[0], *var_seen.shape[1:])


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_cfg_path = os.path.join(os.path.dirname(args.config), "default.yaml")
    cfg = load_config(args.config, default_path=default_cfg_path)

    # Phase A + B
    stats = extract_trainval_stats(cfg, device)
    shrunk = shrink_variances(stats)
    print(f"[Phase B] shrunk variances ready "
          f"(cls_var mean={shrunk['cls_var'].mean().item():.4f})")

    # Frozen phi for ALL classes, re-indexed to the seen/unseen row order of
    # the stats (phi rows are GLOBAL class ids).
    phi = load_semantic_embeddings(cfg.data.semantic_embedding_path).float()
    n_all = len(stats["seen_ids"]) + len(stats["unseen_ids"])
    if phi.shape[0] != n_all:
        raise ValueError(
            f"phi has {phi.shape[0]} classes but dataset has {n_all}.")
    phi_seen = phi[stats["seen_ids"]]
    phi_unseen = phi[stats["unseen_ids"]]

    # Phase C — estimator selection + gate
    best, grid, hold_rows, gate = select_estimator(args, phi_seen, stats)
    if not gate:
        print("[STOP] Bank-quality gate FAILED — nothing saved, no train "
              "budget spent. Inspect the grid above (or adjust nw_h / "
              "ridge_lambda).")
        sys.exit(1)

    # Phase D — refit on all 150 seen, transfer to the 50 unseen.
    var_kind, var_params = best["kind"], dict(best["params"])
    if best["kind"] == "ridge":
        nw_grid = [g for g in grid if g["kind"] == "nw"]
        if not nw_grid:
            raise RuntimeError("Ridge winner but the NW grid is empty.")
        var_params = dict(max(nw_grid, key=lambda g: g["nn_top1"])["params"])
        var_kind = "nw"
        print(f"[Phase D] ridge winner -> variance transfer via NW "
              f"h={var_params['h']}")

    cls_mean_u = transfer_mu(best["kind"], best["params"],
                             phi_seen, phi_unseen, stats["cls_mean"])
    gap_mean_u = transfer_mu(best["kind"], best["params"],
                             phi_seen, phi_unseen, stats["gap_mean"])
    tok_mean_u = transfer_mu(best["kind"], best["params"],
                             phi_seen, phi_unseen, stats["tok_mean_h"])
    cls_var_u = transfer_var(var_kind, var_params,
                             phi_seen, phi_unseen, shrunk["cls_var"])
    gap_var_u = transfer_var(var_kind, var_params,
                             phi_seen, phi_unseen, shrunk["gap_var"])

    cls_std_u = cls_var_u.clamp(min=1e-8).sqrt()
    gap_std_u = gap_var_u.clamp(min=1e-8).sqrt()
    if args.std_scale != 1.0:
        print(f"[Phase D] std_scale={args.std_scale} applied to cls/gap std")
        cls_std_u = cls_std_u * args.std_scale
        gap_std_u = gap_std_u * args.std_scale

    out = {
        "version": "v12_synbank_v1",
        "dataset": cfg.data.dataset,
        "backbone": getattr(cfg.model.vision, "backbone_type", ""),
        "estimator": {
            "kind": best["kind"],
            "params": best["params"],
            "holdout_metrics": {
                "cos_mean": best["cos_mean"],
                "nn_top1": best["nn_top1"],
            },
            "grid": grid,
            "var_transfer": {"kind": var_kind, "params": var_params},
            "std_scale": args.std_scale,
            "seed": args.seed,
        },
        "holdout_seen_ids": [stats["seen_ids"][r] for r in hold_rows],
        "seen_classes": stats["seen_ids"],
        "unseen_classes": stats["unseen_ids"],
        "cls_mean": cls_mean_u,
        "cls_std": cls_std_u,
        "gap_mean": gap_mean_u,
        "gap_std": gap_std_u,
        "tok_mean_h": tok_mean_u,
        "counts_seen": {cid: int(stats["counts"][i].item())
                        for i, cid in enumerate(stats["seen_ids"])},
    }
    for key in ("cls_mean", "cls_std", "gap_mean", "gap_std", "tok_mean_h"):
        if not torch.isfinite(out[key]).all():
            raise RuntimeError(f"Non-finite values in transferred '{key}'.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    mb = out_path.stat().st_size / 1e6
    print(f"[Phase D] saved bank -> {out_path} ({mb:.1f} MB)")
    print(f"  unseen classes: {len(out['unseen_classes'])}, "
          f"cls_mean {tuple(out['cls_mean'].shape)}, "
          f"tok_mean_h {tuple(out['tok_mean_h'].shape)}")


if __name__ == "__main__":
    main()
