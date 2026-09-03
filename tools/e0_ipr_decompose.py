#!/usr/bin/env python
"""E0: Phan ra IPR residual tren checkpoint v10 (u-collapse).

Phân rã p_hat = p_sem + lambda_r * gate * (M_class V) thành các biến thể
counterfactual và đo T1/u/s/H trên checkpoint cuối của run v10 A0
(checkpoints/cub_clip_v10_ca_b16/CUB/final.pth):

  V0  semantic floor : scores = msh(v_tilde, p_sem) (bỏ IPR hoàn toàn —
                       đúng đường out.scores_sem của model)
  V1  full IPR       : p_hat như training (sanity: phải tái hiện u~29.9)
  V2  CFG post-hoc   : p_sem + lambda_r*(R_i - R_null), null = mean-image
                       condition (mean tokens + mean v_tilde trên trainval) —
                       chỉ giữ phần image-conditioned của residual
  V3  constant-only  : p_sem + lambda_r*R_null — chỉ phần constant, không
                       có thành phần image-conditioned (đo xem constant
                       một mình đã đủ gây sập u chưa)
  V4  mean-centered  : p_sem + lambda_r*(R_i - R_bar), R_bar = mean R_i
                       trên subsample trainval (empirical)

Lưu ý: r(∅) theo nghĩa "zero tokens" là degenerate (W_k/W_vca không bias =>
K=V=0 => residual=0 => trừ 0). Null condition ở đây là mean visual condition
trên trainval — hằng số induction có thể tính cả khi train.

Hình học per-class-column: ||lambda_r*R_null||, ||lambda_r*R_bar|| (seen vs
unseen), tangential fraction (sin góc) của prototype CFG-cleaned, drift-to-seen.

E0c: sweep alpha * W_multi (alpha in {0, 0.1, ..., 1.0}) — scale cả weight
và bias của mfa.W_multi, chạy full GZSL eval (V0/V1) từng alpha (alpha=0
~ R6b cls-only; alpha=1 phải khớp kết quả main).

Chạy trên IBM (không train, chỉ forward):
  CUDA_VISIBLE_DEVICES=0 python tools/e0_ipr_decompose.py \
      --config config/cub_clip_v10_ca_b16.yaml \
      --checkpoint checkpoints/cub_clip_v10_ca_b16/CUB/final.pth \
      --out .scratch/e0_v10_a0.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_script_dir, ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from datasets.prepared_dataset import build_prepared_zsl_splits
from model.hsprn_model import HSPRN
from tools.evaluator import ZSLEvaluator
from tools.semantic_policy import configure_semantic_input, resolve_semantic_source
from tools.utils import load_checkpoint, load_config

ALL_VARIANTS = ("V0", "V1", "V2", "V3", "V4")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="config/cub_clip_v10_ca_b16.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/cub_clip_v10_ca_b16/CUB/final.pth")
    parser.add_argument("--out", type=str, default=".scratch/e0_v10_a0.json")
    parser.add_argument("--train_batches", type=int, default=1000,
                        help="Số batch trainval tối đa dùng ước lượng "
                             "R_null/R_bar (mặc định: toàn bộ trainval)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--skip_e0c", action="store_true")
    return parser.parse_args()


# ------------------------------------------------------------------
# IPR residual tái hiện đúng CrossAttentionPrototypeRefiner.forward
# (protypes.py): trả về gate * residual [B,C,d] — phần được nhân lambda_r
# cộng vào p_sem trong p_hat = p + lambda_r * gate * residual.
# ------------------------------------------------------------------
def ipr_residual(ipr, p_sem, v_tilde, tokens):
    batch_size = v_tilde.shape[0]
    class_count = p_sem.shape[0]
    p_expand = p_sem.unsqueeze(0).expand(batch_size, class_count, -1)

    scale = ipr.prototype_dim ** -0.5
    K = ipr.W_k(tokens)
    V = ipr.W_vca(tokens)

    Q_c = ipr.W_qc(p_sem)
    logits_c = torch.einsum("cd,bnd->bcn", Q_c, K) * scale
    M_class = logits_c.softmax(dim=-1)
    residual = M_class @ V

    v_expand = v_tilde.unsqueeze(1).expand(batch_size, class_count, -1)
    gate_input = torch.cat(
        [v_expand, p_expand, v_expand * p_expand], dim=-1
    )
    gate = torch.sigmoid(ipr.Wr(gate_input))
    return gate * residual


def forward_variants(model, p_sem, images, lambda_r, R_null, R_bar,
                     names=ALL_VARIANTS):
    """Scores [B, C_total] cho các biến thể trong `names` trên một batch."""
    with torch.no_grad():
        visual_levels = model.vis_encoder(images)
        v_tilde = model.mfa(visual_levels)
        tokens_h = visual_levels["tokens"][-1]

        B = v_tilde.shape[0]
        p_exp = p_sem.unsqueeze(0).expand(B, -1, -1)
        r_null = R_null.unsqueeze(0)   # [1,C,d] broadcast
        r_bar = R_bar.unsqueeze(0)

        out = {}
        if "V0" in names:
            # p_sem đã unit-norm nên rp_bank(p_exp) == p_exp; giống hệt đường
            # scores_sem trong HSPRN.forward.
            out["V0"] = model.msh(v_tilde, p_exp)
        needs_g = any(v in names for v in ("V1", "V2", "V4"))
        g_res = None
        if needs_g:
            g_res = ipr_residual(model.ipr, p_sem, v_tilde, tokens_h)
        if "V1" in names:
            out["V1"] = model.msh(
                v_tilde, model.rp_bank(p_exp + lambda_r * g_res))
        if "V2" in names:
            # CFG: chỉ giữ phần lệch khỏi phản ứng null
            out["V2"] = model.msh(
                v_tilde, model.rp_bank(p_exp + lambda_r * (g_res - r_null)))
        if "V3" in names:
            # constant-only: phản ứng null thay cho residual theo ảnh
            out["V3"] = model.msh(
                v_tilde, model.rp_bank(p_exp + lambda_r * r_null))
        if "V4" in names:
            out["V4"] = model.msh(
                v_tilde, model.rp_bank(p_exp + lambda_r * (g_res - r_bar)))
        return out


def collect_scores(model, loader, p_sem, lambda_r, R_null, R_bar, device,
                   names=ALL_VARIANTS):
    """Trả về ({variant: scores [N, C_total]}, labels [N])."""
    scores = {v: [] for v in names}
    labels_all = []
    model.eval()
    for batch in loader:
        images, labels = batch[:2]
        images = images.to(device)
        out = forward_variants(model, p_sem, images, lambda_r, R_null,
                               R_bar, names=names)
        for v in names:
            scores[v].append(out[v].cpu())
        labels_all.append(labels)
    return ({v: torch.cat(t, dim=0) for v, t in scores.items()},
            torch.cat(labels_all, dim=0))


def sanity_check_hand(model, attr_full, p_sem, loader, lambda_r, R_null,
                      R_bar, device):
    """Batch đầu test_unseen: đối chiếu hand-recompute với forward thật.
    V1 phải khớp out.scores (u~29.9 của training log), V0 khớp out.scores_sem."""
    batch = next(iter(loader))
    images, _ = batch[:2]
    images = images.to(device)
    with torch.no_grad():
        out = model(attr_full, images)
        hand = forward_variants(model, p_sem, images, lambda_r, R_null,
                                R_bar, names=("V0", "V1"))
    d1 = (hand["V1"] - out.scores).abs().max().item()
    d0 = (hand["V0"] - out.scores_sem).abs().max().item()
    print(f"[E0] sanity hand-vs-model: max|V1-out.scores|={d1:.2e}  "
          f"max|V0-out.scores_sem|={d0:.2e}")
    if d1 > 1e-4 or d0 > 1e-4:
        print("[E0] CẢNH BÁO: hand-recompute lệch model > 1e-4 — "
              "kiểm tra lại ipr_residual trước khi tin kết quả!")


def eval_variant(evaluator, name, s_seen, s_unseen, labels_seen, labels_unseen):
    T1 = evaluator.eval_zsl(s_unseen, labels_unseen)
    u, s, H = evaluator.eval_gzsl(s_seen, labels_seen, s_unseen, labels_unseen)
    cal_u, cal_s, cal_H, cal_gamma = evaluator.eval_gzsl_calibrated(
        s_seen, labels_seen, s_unseen, labels_unseen)
    hub = evaluator.hubness_stats(s_seen, s_unseen)
    print(f"  {name:4s} T1={100*T1:5.2f}  u={100*u:5.2f}  s={100*s:5.2f}  "
          f"H={100*H:5.2f}  cal_H={100*cal_H:5.2f} (γ={cal_gamma:.2f})  "
          f"hub_s/h={hub['hub_seen_ratio']:.1f}/{hub['hub_unseen_ratio']:.1f}")
    return {
        "T1": float(100 * T1), "u": float(100 * u),
        "s": float(100 * s), "H": float(100 * H),
        "cal_u": float(100 * cal_u), "cal_s": float(100 * cal_s),
        "cal_H": float(100 * cal_H), "cal_gamma": float(cal_gamma),
        "hub_seen_ratio": float(hub["hub_seen_ratio"]),
        "hub_unseen_ratio": float(hub["hub_unseen_ratio"]),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[E0] device={device}")

    # ------------------ 1. Config + data (theo train.py) ------------------
    default_cfg_path = os.path.join(os.path.dirname(args.config), "default.yaml")
    cfg = load_config(args.config, default_path=default_cfg_path)

    prepared_root = getattr(cfg.data, "prepared_root", "./prepared_datasets")
    semantic_source = resolve_semantic_source(cfg)
    image_preprocess = getattr(cfg.data, "image_preprocess", None)
    if image_preprocess is None:
        image_preprocess = (
            "clip" if "clip" in cfg.model.vision.backbone_type.lower()
            else "imagenet"
        )

    splits = build_prepared_zsl_splits(
        prepared_root,
        cfg.data.dataset,
        require_full_semantics=True,
        semantic_embedding_path=getattr(cfg.data, "semantic_embedding_path", None),
        image_size=getattr(cfg.data, "image_size", 224),
        image_preprocess=image_preprocess,
    )
    attribute = splits["attribute"]
    seen_classes = splits["seen_classes"]
    unseen_classes = splits["unseen_classes"]
    all_classes = np.concatenate([seen_classes, unseen_classes])
    n_total = len(all_classes)

    configure_semantic_input(cfg, attribute, semantic_source)
    print(f"[E0] seen={len(seen_classes)} unseen={len(unseen_classes)} "
          f"total={n_total} semantic_source={semantic_source}")

    # ------------------ 2. Model + checkpoint ------------------
    model = HSPRN(cfg)
    ckpt = load_checkpoint(args.checkpoint, model)
    model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    lambda_r = float(model.ipr.lambda_r.detach().cpu())
    print(f"[E0] checkpoint epoch={epoch} lambda_r={lambda_r:.4f}")
    eval_stats = ckpt.get("eval_stats", None)
    if eval_stats:
        print(f"[E0] ckpt eval_stats: {str(eval_stats)[:300]}")

    # attr dict 200 lớp (key = str(class_index), giống trainer);
    # p_sem row i tương ứng class i — cột scores cũng là class i.
    attr_full = {
        str(int(ci)): attribute[int(ci)].to(device)
        for ci in range(attribute.shape[0])
    }
    with torch.no_grad():
        p_sem = model.psc(attr_full)["p_sem_norm"]  # [200, 512]

    # ------------------ 3. Loaders ------------------
    batch_size = args.batch_size or cfg.data.batch_size
    num_workers = (args.num_workers if args.num_workers is not None
                   else getattr(cfg.data, "num_workers", 8))

    def make_loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=num_workers, pin_memory=True)

    loader_seen = make_loader(splits["test_seen"])
    loader_unseen = make_loader(splits["test_unseen"])
    loader_train = make_loader(splits["trainval"], shuffle=True)
    evaluator = ZSLEvaluator(seen_classes, unseen_classes)

    # ------------- 4. Null condition + R_bar trên subsample trainval -------------
    # Một pass duy nhất: tích lũy mean tokens / mean v_tilde (cho R_null) và
    # mean gate*residual (R_bar) trên cùng subsample.
    token_sum = None
    v_sum = None
    n_train = 0
    R_bar = torch.zeros_like(p_sem)
    n_rbar = 0
    bi = -1
    for bi, batch in enumerate(loader_train):
        if bi >= args.train_batches:
            break
        images, _ = batch[:2]
        images = images.to(device)
        with torch.no_grad():
            visual_levels = model.vis_encoder(images)
            v_tilde = model.mfa(visual_levels)
            tokens_h = visual_levels["tokens"][-1]
            if token_sum is None:
                token_sum = torch.zeros_like(tokens_h[0], device=device)
                v_sum = torch.zeros_like(v_tilde[0], device=device)
            token_sum = token_sum + tokens_h.sum(dim=0)
            v_sum = v_sum + v_tilde.sum(dim=0)
            n_train += images.shape[0]
            g_res = ipr_residual(model.ipr, p_sem, v_tilde, tokens_h)
            R_bar += g_res.sum(dim=0)
            n_rbar += v_tilde.shape[0]
    tokens_null = (token_sum / n_train).unsqueeze(0)  # [1, N, 768]
    v_null = (v_sum / n_train).unsqueeze(0)            # [1, 512]
    R_bar /= n_rbar
    print(f"[E0] null condition & R_bar từ {n_train} ảnh trainval "
          f"({bi + 1} batch)")

    with torch.no_grad():
        R_null = ipr_residual(model.ipr, p_sem, v_null, tokens_null).squeeze(0)

    results = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "lambda_r": lambda_r,
        "n_train_null": n_train,
        "variants": {},
    }

    # ------------------ 5. Sanity hand-vs-model ------------------
    sanity_check_hand(model, attr_full, p_sem, loader_unseen, lambda_r,
                      R_null, R_bar, device)

    # ------------------ 6. Eval các biến thể ------------------
    print("[E0] ==== Counterfactual (final.pth, epoch %s) ====" % epoch)
    s_seen, labels_seen = collect_scores(
        model, loader_seen, p_sem, lambda_r, R_null, R_bar, device)
    s_unseen, labels_unseen = collect_scores(
        model, loader_unseen, p_sem, lambda_r, R_null, R_bar, device)

    for v in ALL_VARIANTS:
        results["variants"][v] = eval_variant(
            evaluator, v, s_seen[v], s_unseen[v], labels_seen, labels_unseen)

    # ------------------ 7. Hình học per-class-column ------------------
    print("[E0] ==== Hình học residual theo nhóm lớp ====")
    with torch.no_grad():
        seen_mask = torch.tensor(
            np.isin(np.arange(n_total), seen_classes), device=device)

        const_norm = (lambda_r * R_null).norm(dim=-1)     # [C]
        rbar_norm = (lambda_r * R_bar).norm(dim=-1)

        # prototype CFG-cleaned trung bình: p_sem + λ(R_bar - R_null)
        p_v2_mean = p_sem + lambda_r * (R_bar - R_null)
        p_v2_ref = F.normalize(p_v2_mean, dim=-1)
        dot = (p_v2_ref * p_sem).sum(-1, keepdim=True)
        sin_v2 = (p_v2_ref - dot * p_sem).norm(dim=-1).clamp(0, 1)

        # drift-to-seen: cos(prototype, centroid p_sem seen) trước/sau CFG
        seen_centroid = F.normalize(
            p_sem[seen_mask].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        cos_after = (p_v2_ref * seen_centroid.unsqueeze(0)).sum(-1)
        cos_before = (p_sem * seen_centroid.unsqueeze(0)).sum(-1)
        drift = cos_after - cos_before

        results["geometry"] = {
            "const_norm_mean": {
                "seen": float(const_norm[seen_mask].mean()),
                "unseen": float(const_norm[~seen_mask].mean()),
            },
            "rbar_norm_mean": {
                "seen": float(rbar_norm[seen_mask].mean()),
                "unseen": float(rbar_norm[~seen_mask].mean()),
            },
            "tangential_sin_v2_mean": {
                "seen": float(sin_v2[seen_mask].mean()),
                "unseen": float(sin_v2[~seen_mask].mean()),
            },
            "drift_to_seen_cos": {
                "unseen_mean": float(drift[~seen_mask].mean()),
                "unseen_max": float(drift[~seen_mask].max()),
                "seen_mean": float(drift[seen_mask].mean()),
            },
        }
        for k, v_ in results["geometry"].items():
            print(f"  {k}: {json.dumps({kk: round(vv, 4) for kk, vv in v_.items()})}")

    # ------------------ 8. E0c: sweep alpha * W_multi ------------------
    if not args.skip_e0c:
        print("[E0c] ==== Sweep alpha·W_multi (anchored MFA), V0/V1 ====")
        W = model.mfa.W_multi.weight.data
        Bb = model.mfa.W_multi.bias.data
        W_orig = W.detach().clone()
        B_orig = Bb.detach().clone()
        results["e0c"] = {}
        try:
            for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                          0.6, 0.7, 0.8, 0.9, 1.0]:
                W.copy_(W_orig * alpha)
                Bb.copy_(B_orig * alpha)
                s_seen_a, labels_seen_a = collect_scores(
                    model, loader_seen, p_sem, lambda_r, R_null, R_bar,
                    device, names=("V0", "V1"))
                s_unseen_a, labels_unseen_a = collect_scores(
                    model, loader_unseen, p_sem, lambda_r, R_null, R_bar,
                    device, names=("V0", "V1"))
                print(f"  alpha={alpha:.1f}")
                alpha_res = {}
                for v in ("V0", "V1"):
                    alpha_res[v] = eval_variant(
                        evaluator, v, s_seen_a[v], s_unseen_a[v],
                        labels_seen_a, labels_unseen_a)
                results["e0c"][f"alpha_{alpha:.1f}"] = alpha_res
        finally:
            W.copy_(W_orig)
            Bb.copy_(B_orig)
        print("[E0c] alpha=1.0 phải khớp kết quả V0/V1 ở phần main "
              "(self-check nhất quán).")

    # ------------------ 9. Lưu kết quả ------------------
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[E0] Đã lưu: {args.out}")
    print("[E0] Hoàn tất.")


if __name__ == "__main__":
    main()
