import os
import sys

_script_dir = os.path.dirname(__file__)
_root = os.path.normpath(os.path.join(_script_dir, ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
if __package__ is None:
    __package__ = "model"

import torch
import torch.nn.functional as F

from .aggregator import MultiFeatureAggregator
from .hsprn_outputs import HSPRNOutput
from .matching_head import SematicMatchingHead
from .protypes import (
    CrossAttentionPrototypeRefiner,
    InstancePrototypeRefiner,
    RefinedPrototypeBank,
)
from .sematics.constructor import PriSematicConstructor
from .vision_model import VisionModel


class HSPRN(torch.nn.Module):
    """Hierarchical Semantic Prototype Refinement Network."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        self.psc = PriSematicConstructor(cfg)
        self.vis_encoder = VisionModel(cfg)
        self.mfa = MultiFeatureAggregator(cfg)
        # v10: refiner.mode "cross_attn" switches IPR core to the
        # cross-attention variant (patch-token residual); anything else keeps
        # the legacy InstancePrototypeRefiner for full back-compat.
        refiner_cfg = getattr(cfg.model, "refiner", object())
        if isinstance(refiner_cfg, dict):
            refiner_mode = refiner_cfg.get("mode", "mlp")
        else:
            refiner_mode = getattr(refiner_cfg, "mode", "mlp")
        if refiner_mode == "cross_attn":
            self.ipr = CrossAttentionPrototypeRefiner(cfg)
        else:
            self.ipr = InstancePrototypeRefiner(cfg)
        self.rp_bank = RefinedPrototypeBank(cfg)
        self.msh = SematicMatchingHead(cfg)

        # v14 Super-CLIP: Local-Global Fusion Module
        # W_local chiếu vector thuộc tính vi mô v_local [B, 512] để cộng vào v_global
        # Khởi tạo zero-init (LP-FT anchor) -> Epoch 0: v_enhanced == v_global == normalize(cls)
        d_vis = getattr(cfg.model.aggregator, "out_dim", 512)
        self.W_local = torch.nn.Linear(d_vis, d_vis, bias=False)
        torch.nn.init.zeros_(self.W_local.weight)

        loss_cfg = getattr(cfg.model, "loss", object())
        self.return_full = getattr(loss_cfg, "return_full_output", True)

    def forward(self, text_feature, vision_feature):
        psc_out = self.psc(text_feature)
        p_sem_norm = psc_out["p_sem_norm"]
        phi_t_raw = psc_out["phi_t"]
        p_sem_unnorm = psc_out["p_sem_unnorm"]

        visual_levels = self.vis_encoder(vision_feature)

        attr_pred = None
        tokens_h = None
        cls_anchor = None
        v_local = None
        if isinstance(visual_levels, dict):
            # v10 anchored path: {"tokens": [...], "cls": [B,512]}.
            # MFA (mode "anchored"/"cls_only") consumes the dict; the IPR
            # residual is read from the LAST (deepest) token map.
            # v11: keep the frozen zero-shot anchor (bit-exact CLIP CLS
            # normalized) for L_fm/L_zs — it is the same tensor mfa would
            # return with W_multi == 0.
            cls_anchor = F.normalize(visual_levels["cls"], p=2, dim=-1)
            v_global = self.mfa(visual_levels)
            tokens_h = visual_levels["tokens"][-1]
            ipr_res = self.ipr(p_sem_norm, v_global, tokens_h)
            if len(ipr_res) == 3:
                p_hat, attr_pred, v_local = ipr_res
            else:
                p_hat, attr_pred = ipr_res
            # v14 Super-CLIP Fusion: v_enhanced = normalize(v_global + W_local(v_local))
            if v_local is not None:
                v_tilde = F.normalize(v_global + self.W_local(v_local), p=2, dim=-1)
            else:
                v_tilde = v_global
        else:
            v_tilde = self.mfa(visual_levels)
            p_hat = self.ipr(p_sem_norm, v_tilde)
        p_ref = self.rp_bank(p_hat)

        scores = self.msh(v_tilde, p_ref)
        # APR unseen path: scores against the UNREFINED semantic prototypes.
        # Same matching head (any projection applies identically to both).
        p_sem_expand = p_sem_norm.unsqueeze(0).expand(v_tilde.shape[0], -1, -1)
        scores_sem = self.msh(v_tilde, p_sem_expand)

        if not self.return_full:
            return scores

        return HSPRNOutput(
            scores=scores,
            v_embed=v_tilde,
            p_sem=p_sem_norm,
            p_ref=p_ref,
            phi_t=phi_t_raw,
            p_sem_unnorm=p_sem_unnorm,
            scores_sem=scores_sem,
            attr_pred=attr_pred,
            tokens=tokens_h,
            cls_anchor=cls_anchor,
        )


# if __name__ == "__main__":
#     from omegaconf import OmegaConf

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Device: {device}")

#     BASE = {
#         "model": {
#             "vision": {
#                 "backbone_type": "vit_base_patch16_224",
#                 "pretrained": False,
#             },
#             "semantic": {
#                 "backbone_type": "embedding",
#                 "input_dim": 85,
#                 "shared_dim": 128,
#                 "pretrained": False,
#             },
#             "aggregator": {
#                 "input_dim": 2048,
#                 "out_dim": 128,
#             },
#             "loss": {
#                 "return_full_output": True,
#             },
#         },
#         "feature_level": "multilevel",
#     }

#     cfg = OmegaConf.create(BASE)
#     model = HSPRN(cfg).to(device)
#     model.eval()

#     B, C = 4, 3

#     # ---------- Test 1: Full HSPRNOutput ----------
#     print("=" * 60)
#     print("Test 1: Full output (return_full_output=True)")
#     print("=" * 60)

#     text_feat = {f"c{i}": torch.randn(85).to(device) for i in range(C)}
#     vis_feat = torch.randn(B, 2048).to(device)

#     with torch.no_grad():
#         out = model(text_feat, vis_feat)

#     assert isinstance(out, HSPRNOutput), f"expected HSPRNOutput, got {type(out)}"
#     assert out.scores.shape == (B, C), f"scores: {out.scores.shape}"
#     assert out.v_embed.shape == (B, 128), f"v_embed: {out.v_embed.shape}"
#     assert out.p_sem.shape == (C, 128), f"p_sem: {out.p_sem.shape}"
#     assert out.p_ref.shape == (B, C, 128), f"p_ref: {out.p_ref.shape}"
#     assert out.phi_t.shape == (C, 85), f"phi_t: {out.phi_t.shape}"
#     assert out.p_sem_unnorm.shape == (C, 128), f"p_sem_unnorm: {out.p_sem_unnorm.shape}"

#     assert torch.allclose(out.p_sem.norm(p=2, dim=-1), torch.ones(C, device=device), atol=1e-5)
#     assert torch.allclose(out.v_embed.norm(p=2, dim=-1), torch.ones(B, device=device), atol=1e-5)

#     print("  scores:     ", out.scores.shape, "OK")
#     print("  v_embed:    ", out.v_embed.shape, "OK")
#     print("  p_sem:      ", out.p_sem.shape, "L2 norm OK")
#     print("  p_ref:      ", out.p_ref.shape, "OK")
#     print("  phi_t:      ", out.phi_t.shape, "OK")
#     print("  p_sem_unnorm:", out.p_sem_unnorm.shape, "OK")
#     print("  PASSED\n")

#     # ---------- Test 2: Scores-only ----------
#     print("=" * 60)
#     print("Test 2: Scores-only output (return_full_output=False)")
#     print("=" * 60)

#     cfg2 = OmegaConf.merge(
#         OmegaConf.create(BASE),
#         OmegaConf.create({"model": {"loss": {"return_full_output": False}}}),
#     )
#     model2 = HSPRN(cfg2).to(device)
#     model2.eval()

#     with torch.no_grad():
#         out2 = model2(text_feat, vis_feat)

#     assert isinstance(out2, torch.Tensor), f"expected Tensor, got {type(out2)}"
#     assert out2.shape == (B, C), f"scores: {out2.shape}"
#     print("  PASSED\n")

#     # ---------- Test 3: Single class & sample ----------
#     print("=" * 60)
#     print("Test 3: Single class & single sample")
#     print("=" * 60)

#     text_feat_1 = {"c0": torch.randn(85).to(device)}
#     vis_feat_1 = torch.randn(1, 2048).to(device)

#     with torch.no_grad():
#         out3 = model(text_feat_1, vis_feat_1)

#     assert out3.scores.shape == (1, 1), f"scores: {out3.scores.shape}"
#     assert out3.p_sem.shape == (1, 128), f"p_sem: {out3.p_sem.shape}"
#     assert out3.v_embed.shape == (1, 128), f"v_embed: {out3.v_embed.shape}"
#     print("  PASSED\n")

#     # ---------- Test 4: Parameter sharing across forward passes ----------
#     print("=" * 60)
#     print("Test 4: Parameter sharing (same model, different batches)")
#     print("=" * 60)

#     vis_feat_2 = torch.randn(B * 2, 2048).to(device)
#     with torch.no_grad():
#         out4a = model(text_feat, vis_feat)
#         out4b = model(text_feat, vis_feat_2)

#     assert out4a.scores.shape == (B, C)
#     assert out4b.scores.shape == (B * 2, C)
#     assert out4a.p_sem.shape == out4b.p_sem.shape
#     assert torch.equal(out4a.p_sem, out4b.p_sem)
#     print("  Same p_sem (text-driven, batch-agnostic):", out4a.p_sem.shape, "OK")
#     print("  scores (B, C):", out4a.scores.shape, "(B*2, C):", out4b.scores.shape, "OK")
#     print("  PASSED\n")

#     # ---------- Test 5: Loss integration ----------
#     print("=" * 60)
#     print("Test 5: Full loss integration (use_full=True)")
#     print("=" * 60)

#     from losses.hsprn_loss import HSPRNLoss

#     cfg3 = OmegaConf.merge(
#         OmegaConf.create(BASE),
#         OmegaConf.create({"model": {"loss": {"use_full": True, "temperature": 0.07, "lambda1": 1.0, "lambda2": 0.1, "lambda3": 0.1}}}),
#     )

#     model3 = HSPRN(cfg3).to(device)
#     criterion = HSPRNLoss(cfg3).to(device)

#     with torch.no_grad():
#         out5 = model3(text_feat, vis_feat)
#         labels = torch.randint(0, C, (B,), device=device)
#         total_loss, loss_dict = criterion(
#             out5.scores, labels,
#             v_tilde=out5.v_embed,
#             p_sem=out5.p_sem,
#             p_ref=out5.p_ref,
#         )

#     assert total_loss.numel() == 1, f"total_loss should be scalar, got {total_loss.shape}"
#     assert all(k in loss_dict for k in ["loss_cls", "loss_align", "loss_pres", "total_loss"])
#     print(f"  total_loss: {total_loss.item():.6f}")
#     for k, v in loss_dict.items():
#         print(f"    {k}: {v:.6f}")
#     print("  PASSED\n")

#     # ---------- Summary ----------
#     n_params = sum(p.numel() for p in model.parameters())
#     print(f"Model parameters: {n_params:,}")
#     print(f"\n{'=' * 60}")
#     print("All tests passed!")
