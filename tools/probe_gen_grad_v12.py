#!/usr/bin/env python
"""v12 Phase S1 — probe that L_gen really backpropagates through the LIVE
inference path before any training budget is spent.

Loads the model from the v12 config + the synthetic-unseen bank, samples a
small synthetic sub-batch exactly the way trainer.train_one_epoch does,
computes L_gen = CE(scores_syn/tau, y_syn) ONLY (no other loss term), and
prints the gradient norm of every trainable block on that path:

  mfa.W_multi          (unseen gradient via the visual side, v_tilde_syn)
  psc.Ws.down/up       (via p_all, shared with the real-image IPR call)
  ipr W_qc/W_k/W_qa/W_vca/Wr, lambda_r  (via p_ref_syn / attr_pred_syn)

GATE: every trainable row must be finite AND non-zero. Exit code 1 on
failure. ~1 min on L40S, no dataset pass (phi + bank are precomputed), so
this is safe to run before S2.

Usage (IBM):
    python tools/probe_gen_grad_v12.py \
        --config config/cub_clip_v12_g1_b16.yaml \
        --bank ./pre-trained/cub_v12_synbank_b16.pt
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

from tools.utils import load_config, set_seed

from model.hsprn_model import HSPRN
from losses import HSPRNLoss
from datasets.zsl_dataset import load_semantic_embeddings


def parse_args():
    p = argparse.ArgumentParser(
        description="Probe the v12 L_gen gradient path (trainable blocks)")
    p.add_argument("--config", type=str, required=True,
                   help="v12 config (loss.lambda_gen > 0)")
    p.add_argument("--bank", type=str, required=True,
                   help="Synthetic-unseen bank .pt from build_synthetic_bank.py")
    p.add_argument("--n_syn", type=int, default=8,
                   help="Synthetic samples in the probe batch")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_cfg_path = os.path.join(os.path.dirname(args.config), "default.yaml")
    cfg = load_config(args.config, default_path=default_cfg_path)

    loss_cfg = getattr(cfg.model, "loss", object())
    lambda_gen = float(getattr(loss_cfg, "lambda_gen", 0.0))
    if lambda_gen <= 0:
        raise ValueError("probe expects model.loss.lambda_gen > 0")
    gen_samples = int(getattr(loss_cfg, "gen_samples_per_step", 64))

    # --- model + criterion + phi bank (same construction order as train.py).
    model = HSPRN(cfg).to(device)
    criterion = HSPRNLoss(cfg)

    # --- synthetic bank (same validation as Trainer.__init__).
    bank = torch.load(args.bank, map_location="cpu")
    n_unseen = len(bank["unseen_classes"])
    for key, shape in {
        "cls_mean": (n_unseen, 512), "cls_std": (n_unseen, 512),
        "gap_mean": (n_unseen, 3, 768), "gap_std": (n_unseen, 3, 768),
        "tok_mean_h": (n_unseen, 197, 768),
    }.items():
        if tuple(bank[key].shape) != shape:
            raise ValueError(f"bank '{key}' shape {tuple(bank[key].shape)} != {shape}")
    cls_mean = bank["cls_mean"].to(device)
    cls_std = bank["cls_std"].to(device)
    gap_mean = bank["gap_mean"].to(device)
    gap_std = bank["gap_std"].to(device)
    tok_mean_h = bank["tok_mean_h"].to(device)
    unseen_global_ids = torch.as_tensor(
        [int(c) for c in bank["unseen_classes"]], dtype=torch.long, device=device)

    # --- full 200-class semantic bank: phi -> Ws -> normalize (the exact
    # need_full_bank block of trainer.train_one_epoch).
    phi_all = load_semantic_embeddings(
        cfg.data.semantic_embedding_path).float().to(device)
    p_all = torch.nn.functional.normalize(
        model.psc.project_phi(
            phi_all, torch.arange(phi_all.shape[0], device=device)),
        dim=-1,
    )

    # --- sample the synthetic sub-batch exactly like train_one_epoch.
    b = args.n_syn
    idx = torch.randint(n_unseen, (b,), device=device)
    y_syn = unseen_global_ids[idx]  # GLOBAL ids (p_all columns)
    eps_c = torch.randn(b, 512, device=device)
    eps_g = torch.randn(b, 2, 768, device=device)
    cls_syn = cls_mean[idx] + cls_std[idx] * eps_c
    gap_lm = gap_mean[idx][:, :2] + gap_std[idx][:, :2] * eps_g
    tok_syn = tok_mean_h[idx]  # no noise (Run 1)
    visual_levels_syn = {
        "tokens": [gap_lm[:, 0].unsqueeze(1).expand(-1, 197, -1),
                   gap_lm[:, 1].unsqueeze(1).expand(-1, 197, -1),
                   tok_syn],
        "cls": cls_syn,
    }

    # --- forward through the LIVE inference path, L_gen only.
    model.train()
    v_tilde_syn = model.mfa(visual_levels_syn)
    p_hat_syn, attr_pred_syn = model.ipr(p_all, v_tilde_syn, tok_syn)
    p_ref_syn = model.rp_bank(p_hat_syn)
    scores_syn = model.msh(v_tilde_syn, p_ref_syn)
    l_gen = torch.nn.functional.cross_entropy(
        scores_syn / criterion.tau, y_syn)

    # Attention term (same formula as the trainer) so W_qa also receives a
    # gradient — exactly what training would do. GT attributes come from the
    # prepared metadata; skip gracefully when absent.
    attn_contrib = None
    if attr_pred_syn is not None and criterion.lambda_attn > 0:
        gt_path = os.path.join(
            getattr(cfg.data, "prepared_root", "./prepared_datasets"),
            str(cfg.data.dataset), "metadata", "attributes.npy")
        if os.path.exists(gt_path):
            gt = torch.from_numpy(np.load(gt_path)).float().to(device)
            l_attn = torch.nn.functional.mse_loss(
                attr_pred_syn,
                torch.nn.functional.normalize(gt[y_syn], dim=-1))
            attn_contrib = criterion.lambda_attn * l_attn

    loss = lambda_gen * l_gen + (attn_contrib if attn_contrib is not None else 0.0)
    model.zero_grad(set_to_none=True)
    loss.backward()

    # --- gradient report. Leaf parameters only: LowRankWs.weight is a
    # computed property (I + B A) — its grads live on down/up.
    def grad_rows():
        rows = []
        mfa = model.mfa
        if hasattr(mfa, "W_multi"):
            rows.append(("mfa.W_multi.weight", mfa.W_multi.weight))
            rows.append(("mfa.W_multi.bias", mfa.W_multi.bias))
        psc = model.psc
        if hasattr(psc, "Ws"):
            ws = psc.Ws
            if hasattr(ws, "down"):  # LowRankWs: W_s = I + B A
                rows.append(("psc.Ws.down.weight (A)", ws.down.weight))
                rows.append(("psc.Ws.up.weight (B)", ws.up.weight))
            else:
                rows.append(("psc.Ws.weight", ws.weight))
        ipr = model.ipr
        for name in ("W_qc", "W_qa", "W_k", "W_vca", "Wr"):
            mod = getattr(ipr, name, None)
            if mod is not None:
                rows.append((f"ipr.{name}.weight", mod.weight))
        rows.append(("ipr.lambda_r", ipr.lambda_r))
        return rows

    print(f"L_gen (probe, n_syn={b}): {l_gen.item():.4f}  "
          f"[random-init model -> expect ~ln({p_all.shape[0]})="
          f"{float(np.log(p_all.shape[0])):.3f}]")
    if attn_contrib is not None:
        print(f"L_gen_attn contribution: {attn_contrib.item():.4f}")
    else:
        print("L_gen_attn: skipped (no attributes.npy on disk, or "
              "attr_pred is None)")

    print(f"\n{'param':34s} {'trainable':>9s} {'grad-norm':>12s}  verdict")
    ok = True
    for name, p in grad_rows():
        if not p.requires_grad:
            print(f"{name:34s} {str(p.requires_grad):>9s} {'-':>12s}  "
                  "FROZEN (unexpected)")
            ok = False
            continue
        g = p.grad
        if g is None:
            print(f"{name:34s} {str(p.requires_grad):>9s} {'None':>12s}  "
                  "FAIL (no grad)")
            ok = False
            continue
        gn = g.norm().item()
        finite = bool(torch.isfinite(g).all().item())
        nonzero = gn > 0.0
        verdict = "OK" if (finite and nonzero) else "FAIL"
        if not (finite and nonzero):
            ok = False
        print(f"{name:34s} {str(p.requires_grad):>9s} {gn:12.4e}  {verdict}")

    print(f"\nConfig gen_samples_per_step: {gen_samples} "
          f"(probe used {b}; only affects speed)")
    print("GATE:", "PASS — unseen gradient flows through the live path "
          "(W_multi -> v_tilde -> IPR -> rp_bank -> scores)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
