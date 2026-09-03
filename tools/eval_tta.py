#!/usr/bin/env python
"""v9.6 E5: TRANSDUCTIVE test-time adaptation for HSPRN — reported in a
SEPARATE table (uses test-set structure; NOT comparable to inductive rows).

Implements two published mechanisms on top of the frozen inductive scores:
  1. Visual-centroid prototype evolution (DPE-style, Zhang et al. NeurIPS 2024):
     m_c = sum_i q_ic * v_i / sum_i q_ic  (soft class centroids from test
     features), then s'(x,c) = lambda * cos(x, m_c) + (1-lambda) * s(x,c).
  2. Soft-assignment refinement (TransCLIP-style, Zanella et al. NeurIPS 2024):
     iterate q = softmax(s'/tau), recompute centroids, k epochs of BMM-like
     updates; a KL anchor to the initial text scores keeps the text-encoder
     knowledge as a prior (implemented as the (1-lambda) mixture term).

No labels are used at any point. Usage mirrors eval_checkpoint.py:

    python tools/eval_tta.py --config config/<cfg>.yaml \
        --checkpoint checkpoints/<run>/CUB/final.pth \
        --name r6b/tta --lambda_mix 0.5 --tau 0.07 --iters 10
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.local_pkg import import_local_datasets
from tools.utils import load_config, set_seed
from tools.semantic_policy import configure_semantic_input, resolve_semantic_source

_ds = import_local_datasets()
build_prepared_zsl_splits = _ds.build_prepared_zsl_splits

from model.hsprn_model import HSPRN
from tools.evaluator import ZSLEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="Transductive TTA eval (SEPARATE protocol)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prepared_root", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--score_bank", choices=("semantic", "refined"), default="semantic")
    p.add_argument("--phi_path", type=str, default=None)
    p.add_argument("--lambda_mix", type=float, default=0.5,
                   help="weight on the visual-centroid score: "
                        "s' = lam*cos(x,m_c) + (1-lam)*s_text(x,c).")
    p.add_argument("--tau", type=float, default=0.07,
                   help="softmax temperature for the soft assignment q.")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--eval_iters", type=str, default="0",
                   help="comma-separated iteration checkpoints to report, e.g. '0,5,10'.")
    return p.parse_args()


@torch.no_grad()
def collect_embeddings(model, loader, phi_cache, all_classes, num_total_classes,
                       device, score_bank):
    """Return (v_embed [N,D], scores [N,C_total], labels) for a test loader.

    Scores are expanded to the full C_total column space exactly like
    eval_checkpoint.compute_scores (column int(c) = class id c, other
    columns -inf), so downstream evaluators can mask by global class id.
    """
    model.eval()
    attr = {str(int(c)): phi_cache[int(c)] for c in all_classes}
    vs, ss, ys = [], [], []
    for batch in loader:
        features, labels = batch[:2]
        features = features.to(device)
        out = model(attr, features)
        v = F.normalize(out.v_embed, p=2, dim=-1)
        src = out.scores_sem if score_bank == "semantic" else (
            out.scores if hasattr(out, "scores") else out)
        full = torch.full(
            (features.size(0), num_total_classes), -float("inf"), device=device)
        for i, c in enumerate(all_classes):
            full[:, int(c)] = src[:, i]
        ss.append(full.cpu())
        vs.append(v.cpu())
        ys.append(labels)
    return torch.cat(vs), torch.cat(ss), torch.cat(ys)


def dpe_transductive(v, s_text, tau, lam, iters, eval_at):
    """Iterative soft-assignment + centroid evolution. All tensors CPU.

    v: [N,D] normalized test embeddings (seen AND unseen test samples).
    s_text: [N,C_total] inductive text scores (cols outside union are -inf;
    we work only on the union columns).
    """
    C = s_text.shape[1]
    finite_cols = torch.isfinite(s_text.any(dim=0))
    col_idx = torch.nonzero(finite_cols).squeeze(-1)
    s0 = s_text[:, col_idx]                                   # [N,C_loc]
    s0 = torch.where(torch.isfinite(s0), s0, torch.zeros_like(s0))
    s_cur = s0.clone()
    out = {}
    for it in range(iters + 1):
        if it in eval_at:
            full = s_text.clone()
            full[:, col_idx] = s_cur
            out[it] = full
        if it == iters:
            break
        q = F.softmax(s_cur / tau, dim=-1)                    # [N,C_loc]
        denom = q.sum(dim=0).clamp(min=1e-6)                  # [C_loc]
        m = (q.unsqueeze(-1) * v.unsqueeze(1)).sum(dim=0) / denom.unsqueeze(-1)
        m = F.normalize(m, p=2, dim=-1)                       # [C_loc,D]
        s_vis = v @ m.T                                       # [N,C_loc]
        s_cur = lam * s_vis + (1.0 - lam) * s0
    return out


def main():
    args = parse_args()
    default_cfg_path = os.path.join(os.path.dirname(args.config), "default.yaml")
    cfg = load_config(args.config, default_path=default_cfg_path)
    set_seed(cfg.training.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    data_source = str(getattr(cfg.data, "source", "legacy")).lower()
    if data_source != "prepared":
        raise ValueError("eval_tta.py supports data.source='prepared' only.")
    prepared_root = args.prepared_root or getattr(cfg.data, "prepared_root", "./prepared_datasets")
    dataset_name = cfg.data.dataset
    semantic_embedding_path = getattr(cfg.data, "semantic_embedding_path", None)
    semantic_source = resolve_semantic_source(cfg)
    cfg.data.semantic_source = semantic_source
    if semantic_source != "embedding":
        raise ValueError("eval_tta.py expects precomputed phi (semantic_source=embedding).")

    splits = build_prepared_zsl_splits(
        prepared_root, dataset_name,
        require_full_semantics=True,
        semantic_embedding_path=semantic_embedding_path,
        image_size=getattr(cfg.data, "image_size", 224),
        image_preprocess=getattr(cfg.data, "image_preprocess", None) or "clip",
    )
    test_seen, test_unseen = splits["test_seen"], splits["test_unseen"]
    attribute = splits["attribute"]
    seen_classes, unseen_classes = splits["seen_classes"], splits["unseen_classes"]
    configure_semantic_input(cfg, attribute, semantic_source)

    model = HSPRN(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    ckpt_epoch = ckpt.get("epoch", -1) if isinstance(ckpt, dict) else -1

    bs = getattr(cfg.data, "batch_size", 64)
    nw = getattr(cfg.data, "num_workers", 4)
    seen_loader = DataLoader(test_seen, batch_size=bs * 2, shuffle=False,
                             num_workers=nw, pin_memory=device == "cuda")
    unseen_loader = DataLoader(test_unseen, batch_size=bs * 2, shuffle=False,
                               num_workers=nw, pin_memory=device == "cuda")
    all_classes = list(map(int, seen_classes)) + list(map(int, unseen_classes))
    if args.phi_path and args.phi_path != semantic_embedding_path:
        obj = torch.load(args.phi_path, map_location="cpu")
        phi_cache = obj["embeddings"].detach().float().to(device)
    else:
        phi_cache = attribute.detach().to(device)

    num_total = int(attribute.shape[0])
    v_s, s_s, y_s = collect_embeddings(model, seen_loader, phi_cache, all_classes,
                                       num_total, device, args.score_bank)
    v_u, s_u, y_u = collect_embeddings(model, unseen_loader, phi_cache, all_classes,
                                       num_total, device, args.score_bank)
    v_all = torch.cat([v_s, v_u], dim=0)
    s_all = torch.cat([s_s, s_u], dim=0)

    eval_at = set(int(x) for x in args.eval_iters.split(",") if x.strip())
    snapshots = dpe_transductive(v_all, s_all, args.tau, args.lambda_mix,
                                 args.iters, eval_at)

    evaluator = ZSLEvaluator(seen_classes, unseen_classes)
    n_seen = v_s.shape[0]
    name = args.name or Path(args.checkpoint).parent.parent.name
    for it in sorted(snapshots):
        full = snapshots[it]
        sc_s, sc_u = full[:n_seen], full[n_seen:]
        t1 = evaluator.eval_zsl(sc_u, y_u) * 100
        u, s, H = evaluator.eval_gzsl(sc_s, y_s, sc_u, y_u)
        cu, cs, cH, g = evaluator.eval_gzsl_calibrated(sc_s, y_s, sc_u, y_u)
        print(f"TTA_RESULT {name} epoch={ckpt_epoch} bank={args.score_bank} "
              f"iter={it} lam={args.lambda_mix} tau={args.tau} "
              f"T1_unseen={t1:.1f} u={u * 100:.1f} s={s * 100:.1f} H={H * 100:.1f} "
              f"cal_u={cu * 100:.1f} cal_s={cs * 100:.1f} cal_H={cH * 100:.1f} "
              f"gamma={g:.2f}")


if __name__ == "__main__":
    main()
