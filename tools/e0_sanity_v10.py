#!/usr/bin/env python
"""E0 sanity checks for HSPRN-v10 (run on IBM, conda py310).

Order of checks (plan item E0; the v10 A0 config must build, so
`precompute_attribute_queries.py` must have run BEFORE this script):

  1. Metadata: attributes.npy / attribute_names.json for all 4 prepared
     datasets (expected n_attr: CUB 312 / APY 64 / AWA2 85 / SUN 102).
  2. extract_levels_anchored: tokens [B,197,768] x3 + cls [B,512], and
     cls matches the standard backbone(x) CLIP CLS to < 1e-5.
  3. Bit-exact epoch-0 anchor: fresh v10 (A0 cfg) vs fresh R6b baseline on
     the same images + phi: max|d scores| and max|d scores_sem| < 1e-5.
     This is the entry condition of the whole v10 plan.
  4. Loss + optimizer steps: L_attn present in loss_dict, finite, > 0;
     decreases over 10 steps on one train batch; labels index the GT
     attribute matrix (global ids) and rows look right (CUB ~15 active).

Usage:
    python tools/e0_sanity_v10.py \
        --baseline config/cub_clip_v9_r6b_noref_v8phi_b16.yaml \
        --v10 config/cub_clip_v10_ca_b16.yaml --device cuda
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
from torch.utils.data import DataLoader

from tools.local_pkg import import_local_datasets
from tools.utils import load_config, set_seed
from tools.semantic_policy import configure_semantic_input, resolve_semantic_source

_ds = import_local_datasets()
build_prepared_zsl_splits = _ds.build_prepared_zsl_splits

from model.hsprn_model import HSPRN
from losses.hsprn_loss import HSPRNLoss

EXPECTED_ATTR = {"CUB": 312, "APY": 64, "AWA2": 85, "SUN": 102}
FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def parse_args():
    p = argparse.ArgumentParser(description="E0 sanity for HSPRN-v10")
    p.add_argument("--baseline", type=str,
                   default="config/cub_clip_v9_r6b_noref_v8phi_b16.yaml")
    p.add_argument("--v10", type=str,
                   default="config/cub_clip_v10_ca_b16.yaml")
    p.add_argument("--prepared_root", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def load_run_cfg(path, prepared_root=None):
    default_cfg_path = os.path.join(os.path.dirname(path), "default.yaml")
    cfg = load_config(path, default_path=default_cfg_path)
    if prepared_root:
        cfg.data.prepared_root = prepared_root
    return cfg


def check_metadata(prepared_root):
    for ds, n_expected in EXPECTED_ATTR.items():
        attr_path = os.path.join(prepared_root, ds, "metadata", "attributes.npy")
        names_path = os.path.join(prepared_root, ds, "metadata", "attribute_names.json")
        if not os.path.exists(attr_path):
            check(f"metadata {ds}", False, f"missing {attr_path}")
            continue
        arr = np.load(attr_path)
        n_names = -1
        if os.path.exists(names_path):
            import json
            with open(names_path, "r", encoding="utf-8") as handle:
                n_names = len(json.load(handle))
        ok = arr.shape[1] == n_expected and n_names == n_expected
        check(
            f"metadata {ds}", ok,
            f"attributes {arr.shape}, names {n_names}, expected n_attr {n_expected}",
        )


def build_model_and_splits(cfg, device):
    semantic_source = resolve_semantic_source(cfg)
    cfg.data.semantic_source = semantic_source
    splits = build_prepared_zsl_splits(
        cfg.data.prepared_root, cfg.data.dataset,
        require_full_semantics=True,
        semantic_embedding_path=cfg.data.semantic_embedding_path,
        image_size=getattr(cfg.data, "image_size", 224),
        image_preprocess=getattr(cfg.data, "image_preprocess", None) or "clip",
    )
    configure_semantic_input(cfg, splits["attribute"], semantic_source)
    model = HSPRN(cfg).to(device)
    model.eval()
    return model, splits


@torch.no_grad()
def check_anchored_extract(model_v10, images, device):
    backbone = model_v10.vis_encoder.backbone
    levels = [4, 8, 12]
    out = backbone.extract_levels_anchored(images.to(device), levels)
    tokens, cls = out["tokens"], out["cls"]
    shapes_ok = (
        len(tokens) == 3
        and all(t.shape[1] == 197 and t.shape[2] == 768 for t in tokens)
        and cls.shape == (images.shape[0], 512)
    )
    check("extract_levels_anchored shapes", shapes_ok,
          f"tokens {[tuple(t.shape) for t in tokens]}, cls {tuple(cls.shape)}")
    std_cls = backbone(images.to(device))
    delta = (cls - std_cls).abs().max().item()
    check("cls == backbone(x)", delta < 1e-5, f"max|d| = {delta:.2e}")
    return tokens, cls, std_cls


@torch.no_grad()
def check_bitexact_anchor(model_v10, model_r6b, splits, device, batch_size):
    phi = splits["attribute"].detach().float().to(device)
    all_classes = (
        list(map(int, splits["seen_classes"]))
        + list(map(int, splits["unseen_classes"]))
    )
    attr = {str(int(c)): phi[int(c)] for c in all_classes}

    loader = DataLoader(splits["test_unseen"], batch_size=batch_size,
                        shuffle=False, num_workers=4)
    images, labels = next(iter(loader))
    images = images.to(device)

    out_v10 = model_v10(attr, images)
    out_r6b = model_r6b(attr, images)

    d_scores = (out_v10.scores - out_r6b.scores).abs().max().item()
    d_sem = (out_v10.scores_sem - out_r6b.scores_sem).abs().max().item()
    d_v = (out_v10.v_embed - out_r6b.v_embed).abs().max().item()
    check("epoch-0 anchor: scores", d_scores < 1e-5, f"max|d| = {d_scores:.2e}")
    check("epoch-0 anchor: scores_sem", d_sem < 1e-5, f"max|d| = {d_sem:.2e}")
    check("epoch-0 anchor: v_embed", d_v < 1e-5, f"max|d| = {d_v:.2e}")
    return images, labels


def check_train_steps(model_v10, cfg, splits, device, batch_size):
    phi = splits["attribute"].detach().float().to(device)
    gt_attr_path = os.path.join(
        cfg.data.prepared_root, cfg.data.dataset, "metadata", "attributes.npy")
    gt_matrix = torch.from_numpy(np.load(gt_attr_path)).float().to(device)
    check("GT attribute matrix rows == classes",
          gt_matrix.shape[0] == phi.shape[0],
          f"gt {tuple(gt_matrix.shape)} vs phi {tuple(phi.shape)}")

    seen = list(map(int, splits["seen_classes"]))
    label_map = {c: i for i, c in enumerate(seen)}
    seen_attr = {str(int(c)): phi[int(c)] for c in seen}

    loader = DataLoader(splits["trainval"], batch_size=batch_size,
                       shuffle=False, num_workers=4)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    mapped = torch.tensor([label_map[int(l)] for l in labels.cpu()],
                          dtype=torch.long, device=device)

    # labels are GLOBAL ids — the GT row of the first label must have roughly
    # the published number of active attributes (CUB ~15), and must differ
    # from another class's row.
    row = gt_matrix[labels[0]]
    n_active = int((row > 0).sum().item())
    other = gt_matrix[(int(labels[0]) + 1) % gt_matrix.shape[0]]
    check("labels <-> GT attribute row", n_active > 0 and not torch.equal(row, other),
          f"label {int(labels[0])}: {n_active} active attributes "
          f"(row {n_active} vs other {(other > 0).sum().item()})")

    model_v10.train()
    criterion = HSPRNLoss(cfg).to(device)
    params = [p for p in model_v10.parameters() if p.requires_grad]
    print(f"  trainable tensors: {len(params)}, "
          f"total numel {sum(p.numel() for p in params):,}")
    opt = torch.optim.Adam(params, lr=1e-3)

    traj = []
    for step in range(10):
        opt.zero_grad()
        out = model_v10(seen_attr, images)
        loss, loss_dict = criterion(
            out.scores, mapped,
            v_tilde=out.v_embed, p_sem=out.p_sem, p_ref=out.p_ref,
            phi_t=out.phi_t, p_sem_unnorm=out.p_sem_unnorm,
            attr_pred=out.attr_pred, gt_attr=gt_matrix[labels],
        )
        loss.backward()
        opt.step()
        traj.append(loss_dict.get("loss_attn"))
    print(f"  loss_attn trajectory (10 steps, same batch): "
          + " ".join(f"{v:.5f}" if v is not None else "None" for v in traj))

    has_attn = traj[0] is not None
    check("L_attn present and > 0", has_attn and traj[0] > 0,
          f"step0 = {traj[0]}")
    finite = all(np.isfinite(v) for v in traj)
    check("L_attn finite", finite)
    decreased = traj[-1] < traj[0]
    check("L_attn decreases over 10 steps", decreased,
          f"{traj[0]:.5f} -> {traj[-1]:.5f}")
    model_v10.eval()


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    cfg_r6b = load_run_cfg(args.baseline, args.prepared_root)
    cfg_v10 = load_run_cfg(args.v10, args.prepared_root)
    # The 10-step check below calls the criterion WITHOUT m_s/p_all/teacher
    # (L_trans needs a full trainval caching pass that only the trainer
    # performs); forcing lambda_trans=0 keeps that check on the anchor-only
    # path. The real L_trans path is exercised at trainer __init__ — verify
    # the "Transfer-alignment ON" log line when training.
    cfg_v10.model.loss.lambda_trans = 0.0
    set_seed(getattr(cfg_v10.training, "seed", 42))
    prepared_root = args.prepared_root or cfg_v10.data.prepared_root

    print(f"=== E0.1: attribute metadata ({prepared_root}) ===")
    check_metadata(prepared_root)

    print("=== E0.2/E0.3: build fresh v10 + R6b models ===")
    model_v10, splits = build_model_and_splits(cfg_v10, device)
    model_r6b, _ = build_model_and_splits(cfg_r6b, device)

    loader = DataLoader(splits["test_unseen"], batch_size=args.batch_size,
                       shuffle=False, num_workers=4)
    images, _ = next(iter(loader))

    print("=== E0.2: extract_levels_anchored vs backbone(x) ===")
    check_anchored_extract(model_v10, images, device)

    print("=== E0.3: bit-exact epoch-0 anchor (v10 vs R6b) ===")
    check_bitexact_anchor(model_v10, model_r6b, splits, device, args.batch_size)

    print("=== E0.4: L_attn + 10 optimizer steps ===")
    check_train_steps(model_v10, cfg_v10, splits, device, args.batch_size)

    print("\n" + ("E0 FAILED: " + ", ".join(FAILURES) if FAILURES
                  else "E0 PASSED — all checks green."))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
