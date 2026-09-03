#!/usr/bin/env python
"""
Evaluate a saved HSPRN checkpoint on ZSL / GZSL metrics (no training).

Reports per checkpoint:
    - T1 unseen  : conventional ZSL Top-1 per-class acc (candidate = unseen only)
    - T1 seen    : seen-only ZSL variant (candidate = seen only)  [NEW]
    - u / s / H  : GZSL raw (candidate = union seen+unseen)
    - Cal u/s/H/γ: GZSL after calibrated stacking (Chao et al., TPAMI 2016)

Usage (on IBM, prepared-embedding config):
    python tools/eval_checkpoint.py \
        --config config/grid_b16pc_combo.yaml \
        --checkpoint checkpoints/grid_b16pc_combo/CUB/best.pth \
        --device cuda
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
    parser = argparse.ArgumentParser(description="Evaluate HSPRN checkpoint (ZSL + GZSL)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prepared_root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--name", type=str, default=None,
                        help="Run label for the RESULT line (default: checkpoint basename dir)")
    parser.add_argument("--score_bank", choices=("refined", "semantic", "semantic_ens"),
                        default="refined",
                        help="B4: 'refined' scores against the instance-refined prototype bank "
                             "(out.scores); 'semantic' scores directly against the semantic "
                             "prototypes (out.scores_sem), bypassing refinement at inference; "
                             "'semantic_ens' (v9.6 E2) additionally replaces each class "
                             "prototype by its per-sentence prototypes and averages the "
                             "per-sentence scores (CuPL-style ensemble).")
    parser.add_argument("--phi_path", type=str, default=None,
                        help="v9.6: override the .pt phi file (must contain 'embeddings'; "
                             "for semantic_ens also 'embeddings_sentences'+'sentence_mask'). "
                             "Lets one config evaluate hybrid/control phi variants.")
    parser.add_argument("--score_correction", choices=("none", "csls"), default="none",
                        help="v9.6 E4: 'csls' applies prototype-side hubness correction "
                             "CSLS(x,c) = 2*cos(x,c) - r(c) with r(c) = mean cos to the "
                             "k nearest OTHER prototypes (Dinu&Baroni ICLR'15; Conneau "
                             "ACL'18). Prototype-side only => fully inductive.")
    parser.add_argument("--csls_k", type=int, default=10,
                        help="k nearest prototypes for the CSLS hub term.")
    parser.add_argument("--csls_beta", type=float, default=1.0,
                        help="weight on the (centered) hub term; negative ADDS "
                             "density to dense prototypes (sign flips because "
                             "Spearman(r, win-count) < 0 on this pipeline).")
    return parser.parse_args()


@torch.no_grad()
def compute_scores(model, loader, phi_cache, class_indices, num_total_classes, device,
                   score_bank="refined", sent_emb=None, sent_mask=None):
    """Scores (N, C_total) over union(class_indices); other columns = -inf.

    score_bank:
      refined       — out.scores (instance-refined bank).
      semantic      — out.scores_sem (unrefined prototypes).
      semantic_ens  — v9.6 E2: per-sentence prototypes Ws(phi_{c,s}) for each
                      class c and sentence s; score = MEAN of per-sentence
                      cosine scores (CuPL-style), then normalized by the
                      sentence count actually present (mask).
    sent_emb/sent_mask: [C_local, S, D] / [C_local, S] tensors matching
    class_indices order (already on device). Only used by semantic_ens.
    """
    model.eval()
    attr = {str(int(c)): phi_cache[int(c)] for c in class_indices}
    scores_list, labels_list = [], []
    for batch in loader:
        features, labels = batch[:2]
        features = features.to(device)
        out = model(attr, features)
        if score_bank == "semantic_ens":
            if sent_emb is None or sent_mask is None:
                raise ValueError("semantic_ens needs sent_emb/sent_mask on the phi file.")
            if not hasattr(out, "p_sem_unnorm"):
                raise ValueError("semantic_ens needs model.loss.return_full_output=true.")
            # Per-sentence prototypes through the same Ws the model uses.
            C_loc, S, D = sent_emb.shape
            if getattr(model.psc, "class_residual", None) is not None:
                raise ValueError(
                    "semantic_ens is not implemented for class_residual configs "
                    "(B3 was rejected by Phase 1 anyway).")
            ws = model.psc.Ws
            flat = sent_emb.reshape(C_loc * S, D)
            protos = ws(flat).reshape(C_loc, S, -1)
            protos = F.normalize(protos, p=2, dim=-1)                # [C,S,D]
            v = F.normalize(out.v_embed, p=2, dim=-1)                # [B,D]
            cos = torch.einsum("csd,bd->bcs", protos, v)             # [B,C,S]
            mask = sent_mask.unsqueeze(0).to(cos.dtype)              # [1,C,S]
            per_class = (cos * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
            scores_subset = per_class                                 # [B,C_local]
        elif score_bank == "semantic":
            if not hasattr(out, "scores_sem"):
                raise ValueError(
                    "--score_bank semantic needs model.loss.return_full_output=true "
                    "so the model also returns scores against the unrefined prototypes.")
            scores_subset = out.scores_sem
        else:
            scores_subset = out.scores if hasattr(out, "scores") else out
        scores_full = torch.full(
            (features.size(0), num_total_classes), -float("inf"), device=device
        )
        for i, c in enumerate(class_indices):
            scores_full[:, int(c)] = scores_subset[:, i]
        scores_list.append(scores_full.cpu())
        labels_list.append(labels)
    return torch.cat(scores_list, dim=0), torch.cat(labels_list, dim=0)


def apply_csls_prototype(scores, proto_cos, k=10, beta=1.0):
    """Prototype-side hubness correction (v9.6 E4), modality-gap aware.

    Raw CSLS (2*cos - r(c), Dinu&Baroni ICLR'15; Conneau ACL'18) is NOT
    directly portable here: CLIP image-proto cosines live near ~0.25 while
    proto-proto cosines near ~0.85 (modality gap, Liang NeurIPS'22), so
    subtracting r(c) at its own scale inverts the score ordering (measured:
    T1 74.4 -> 46.8). The hub signal that matters is r(c) RELATIVE to other
    prototypes, so we subtract the mean-centered hub term:

        s'(x,c) = cos(x,c) - (r(c) - mean(r))

    A prototype denser than average (hub) loses score; a sparse one gains.
    """
    corrected = scores.clone()
    C = proto_cos.shape[0]
    if k >= C - 1:
        k = C - 1
    off = proto_cos.clone()
    off.fill_diagonal_(-1.0)
    topk = off.topk(k, dim=-1).values
    r = topk.mean(dim=-1)                                   # [C]
    r = r - r.mean()          # relative hub-ness (modality-gap aware, see docstring)
    finite = torch.isfinite(scores)
    cols = torch.unique(torch.nonzero(finite.any(dim=0)).squeeze(-1))
    corrected[:, cols] = corrected[:, cols] - beta * r[cols].unsqueeze(0)
    return corrected


def main():
    args = parse_args()

    default_cfg_path = os.path.join(os.path.dirname(args.config), "default.yaml")
    cfg = load_config(args.config, default_path=default_cfg_path)
    set_seed(cfg.training.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Dataset (prepared source only — the v8 precomputed path used by all runs)
    data_source = str(getattr(cfg.data, "source", "legacy")).lower()
    if data_source != "prepared":
        raise ValueError("eval_checkpoint.py supports data.source='prepared' only.")
    prepared_root = args.prepared_root or getattr(cfg.data, "prepared_root", "./prepared_datasets")
    dataset_name = cfg.data.dataset
    semantic_embedding_path = getattr(cfg.data, "semantic_embedding_path", None)

    semantic_source = resolve_semantic_source(cfg)
    cfg.data.semantic_source = semantic_source
    if semantic_source != "embedding":
        raise ValueError(
            f"Expected semantic_source='embedding' (precomputed phi), got {semantic_source!r}."
        )

    image_preprocess = getattr(cfg.data, "image_preprocess", None) or "clip"
    splits = build_prepared_zsl_splits(
        prepared_root,
        dataset_name,
        require_full_semantics=True,
        semantic_embedding_path=semantic_embedding_path,
        image_size=getattr(cfg.data, "image_size", 224),
        image_preprocess=image_preprocess,
    )
    test_seen = splits["test_seen"]
    test_unseen = splits["test_unseen"]
    attribute = splits["attribute"]
    seen_classes = splits["seen_classes"]
    unseen_classes = splits["unseen_classes"]
    configure_semantic_input(cfg, attribute, semantic_source)

    print(f"Test seen: {len(test_seen)} | Test unseen: {len(test_unseen)} | "
          f"seen classes: {len(seen_classes)} | unseen classes: {len(unseen_classes)}")

    # Model + checkpoint
    model = HSPRN(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    # strict=False: CLIP backbones may save auxiliary buffers
    # (e.g. positional_embedding_new from pos-embed interpolation).
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        raise RuntimeError(
            f"Checkpoint is missing required keys: {result.missing_keys[:10]}")
    if result.unexpected_keys:
        print(f"(ignoring {len(result.unexpected_keys)} auxiliary "
              f"checkpoint keys, e.g. {result.unexpected_keys[0]})")
    ckpt_epoch = ckpt.get("epoch", -1) if isinstance(ckpt, dict) else -1
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt_epoch})")

    # Loaders (same settings as trainer: batch_size*2, no shuffle)
    bs = getattr(cfg.data, "batch_size", 64)
    nw = getattr(cfg.data, "num_workers", 4)
    seen_loader = DataLoader(test_seen, batch_size=bs * 2, shuffle=False,
                             num_workers=nw, pin_memory=device == "cuda")
    unseen_loader = DataLoader(test_unseen, batch_size=bs * 2, shuffle=False,
                               num_workers=nw, pin_memory=device == "cuda")

    # v11 Fix 2a: the IPR cfg_mode='cfg' null condition (mean tokens +
    # mean v_tilde over trainval) lives in NON-persistent buffers, so a
    # loaded checkpoint does not carry it — rebuild it here (protocol-safe:
    # trainval only, eval transform, frozen backbone) or the CFG forward
    # silently degenerates to the no-subtraction branch.
    ipr = getattr(model, "ipr", None)
    if ipr is not None and getattr(ipr, "cfg_mode", "legacy") == "cfg" \
            and not getattr(ipr, "_null_filled", False):
        model.eval()
        token_sum = None
        v_sum = None
        n_null = 0
        trainval = splits.get("trainval", None)
        if trainval is None:
            raise RuntimeError(
                "cfg_mode='cfg' eval requires the trainval split to rebuild "
                "the IPR null condition (require_full_semantics splits)."
            )
        null_loader = DataLoader(trainval, batch_size=bs * 2, shuffle=False,
                                 num_workers=nw,
                                 pin_memory=device == "cuda")
        with torch.no_grad():
            for batch in null_loader:
                images = batch[0].to(device)
                visual_levels = model.vis_encoder(images)
                if not isinstance(visual_levels, dict):
                    raise RuntimeError(
                        "cfg_mode='cfg' eval requires the anchored vision path."
                    )
                tokens_h = visual_levels["tokens"][-1]
                v_tilde = model.mfa(visual_levels)
                if token_sum is None:
                    token_sum = torch.zeros_like(tokens_h[0])
                    v_sum = torch.zeros_like(v_tilde[0])
                token_sum += tokens_h.sum(dim=0)
                v_sum += v_tilde.sum(dim=0)
                n_null += images.shape[0]
        if n_null == 0:
            raise RuntimeError("Null-condition rebuild found no trainval batches.")
        ipr.set_null_condition((token_sum / n_null).unsqueeze(0),
                               (v_sum / n_null).unsqueeze(0))
        print(f"IPR CFG null condition rebuilt over {n_null} trainval images.")

    # Frozen phi cache = precomputed attribute (identity psc backbone).
    # v9.6: --phi_path overrides the config's phi file (hybrid/Waffle variants)
    # WITHOUT touching the training config, so the same checkpoint can be
    # scored against several phi sources.
    num_total_classes = int(attribute.shape[0])
    phi_file = args.phi_path or semantic_embedding_path
    if phi_file != semantic_embedding_path or args.score_bank == "semantic_ens":
        obj = torch.load(phi_file, map_location="cpu")
        phi_cache = obj["embeddings"].detach().float().to(device)
        if len(phi_cache) != num_total_classes:
            raise RuntimeError(
                f"phi override has {len(phi_cache)} classes, expected {num_total_classes}")
    else:
        phi_cache = attribute.detach().to(device)
    all_classes = list(map(int, seen_classes)) + list(map(int, unseen_classes))

    sent_emb = sent_mask = None
    if args.score_bank == "semantic_ens":
        if phi_file is None:
            raise ValueError("semantic_ens needs --phi_path to a phi file with "
                             "embeddings_sentences + sentence_mask.")
        obj = torch.load(phi_file, map_location="cpu")
        try:
            se = obj["embeddings_sentences"].float()
            sm = obj["sentence_mask"].bool()
        except KeyError as e:
            raise ValueError(
                f"{phi_file} lacks per-sentence tensors (precompute with "
                f"--export_sentences): {e}")
        # precompute_class_phi stores rows in dataset class_index order, the
        # same order as phi_cache rows. Subset to all_classes so row i matches
        # class_indices[i] inside compute_scores.
        if se.shape[0] != num_total_classes:
            raise RuntimeError(
                f"sentence tensor has {se.shape[0]} classes, expected {num_total_classes}")
        order = torch.tensor(all_classes, dtype=torch.long)
        sent_emb = se[order].to(device)
        sent_mask = sm[order].to(device)

    scores_seen, labels_seen = compute_scores(
        model, seen_loader, phi_cache, all_classes, num_total_classes, device,
        score_bank=args.score_bank, sent_emb=sent_emb, sent_mask=sent_mask)
    scores_unseen, labels_unseen = compute_scores(
        model, unseen_loader, phi_cache, all_classes, num_total_classes, device,
        score_bank=args.score_bank, sent_emb=sent_emb, sent_mask=sent_mask)

    if args.score_correction == "csls":
        protos = []
        with torch.no_grad():
            # Prototypes after Ws, in all_classes order (same bank the
            # semantic/semantic_ens paths score against).
            if args.score_bank in ("semantic", "semantic_ens"):
                ws = model.psc.Ws
                base = phi_cache[torch.tensor(all_classes, device=device)]
                protos = F.normalize(ws(base), p=2, dim=-1)
            else:
                raise ValueError("csls correction is implemented for semantic banks "
                                 "only (refined bank is per-instance).")
            proto_cos = protos @ protos.T
        scores_seen = apply_csls_prototype(scores_seen.to(device), proto_cos,
                                           k=args.csls_k, beta=args.csls_beta).cpu()
        scores_unseen = apply_csls_prototype(scores_unseen.to(device), proto_cos,
                                             k=args.csls_k, beta=args.csls_beta).cpu()

    evaluator = ZSLEvaluator(seen_classes, unseen_classes)
    t1_unseen = evaluator.eval_zsl(scores_unseen, labels_unseen) * 100
    t1_seen = evaluator.eval_zsl_seen(scores_seen, labels_seen) * 100
    u, s, H = evaluator.eval_gzsl(scores_seen, labels_seen, scores_unseen, labels_unseen)
    cal_u, cal_s, cal_H, cal_gamma = evaluator.eval_gzsl_calibrated(
        scores_seen, labels_seen, scores_unseen, labels_unseen)
    hub = evaluator.hubness_stats(scores_seen, scores_unseen)

    name = args.name or Path(args.checkpoint).parent.parent.name
    print(f"RESULT {name} epoch={ckpt_epoch} bank={args.score_bank} "
          f"correction={args.score_correction} k={args.csls_k} "
          f"phi={os.path.basename(str(phi_file))} "
          f"T1_unseen={t1_unseen:.1f} T1_seen={t1_seen:.1f} "
          f"u={u * 100:.1f} s={s * 100:.1f} H={H * 100:.1f} "
          f"cal_u={cal_u * 100:.1f} cal_s={cal_s * 100:.1f} "
          f"cal_H={cal_H * 100:.1f} gamma={cal_gamma:.2f} "
          f"hub_s={hub['hub_seen_ratio']:.2f} hub_u={hub['hub_unseen_ratio']:.2f}")


if __name__ == "__main__":
    main()
