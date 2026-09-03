from collections.abc import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiFeatureAggregator(nn.Module):
    """Multi-level Feature Aggregator for Eqs. (13)-(17).

    The theoretical path consumes {z_l, z_m, z_h}. For the repo's cached
    feature path, a single 2D tensor is treated as already extracted visual
    features and projected into the shared visual-semantic space.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        aggregator_cfg = cfg.model.aggregator
        self.out_dim = getattr(aggregator_cfg, "out_dim", 512)
        self.level_dims = getattr(aggregator_cfg, "level_dims", None)
        if self.level_dims is None:
            self.level_dims = [
                getattr(aggregator_cfg, "dim_l", 768),
                getattr(aggregator_cfg, "dim_m", 768),
                getattr(aggregator_cfg, "dim_h", 768),
            ]
        self.single_dim = getattr(aggregator_cfg, "input_dim", 2048)

        # v10 anchored MFA (Eqs. 14-16 realized as an anchored residual,
        # LP-FT style): the visual encoder returns {"tokens": [...], "cls"}
        # from ONE forward (CLIPVisualBackbone.extract_levels_anchored), and
        #   v_tilde = normalize(cls + W_multi[GAP(z_l); GAP(z_m); GAP(z_h)])
        # W_multi is zero-init => epoch-0 v_tilde == normalize(cls) == the
        # R6b baseline's (identity Wv_single over CLS) output bit-exactly.
        # "cls_only" (A2 ablation): v_tilde = normalize(cls), no W_multi.
        self.mode = str(getattr(aggregator_cfg, "mode", "default")).lower()
        if self.mode not in ("default", "anchored", "cls_only"):
            raise ValueError(
                "model.aggregator.mode must be 'default', 'anchored', or 'cls_only'."
            )
        self._wv_anchor_state = None
        if self.mode in ("anchored", "cls_only"):
            if self.mode == "anchored":
                self.W_multi = nn.Linear(sum(self.level_dims), self.out_dim, bias=True)
                nn.init.zeros_(self.W_multi.weight)
                nn.init.zeros_(self.W_multi.bias)
            return  # no legacy Wv/Wv_single machinery on the v10 path

        self.pyramid = bool(getattr(aggregator_cfg, "pyramid", False))
        self.level_proj_dim = int(getattr(aggregator_cfg, "level_dim", self.out_dim))
        if self.pyramid:
            self.level_proj = nn.ModuleList([
                nn.Linear(dim, self.level_proj_dim, bias=False)
                for dim in self.level_dims
            ])
            self.Wv = nn.Linear(
                len(self.level_dims) * self.level_proj_dim, self.out_dim, bias=False
            )
        else:
            self.level_proj = None
            self.Wv = nn.Linear(sum(self.level_dims), self.out_dim, bias=False)
        self.Wv_single = nn.Linear(self.single_dim, self.out_dim, bias=False)

        # Aligned-geometry countermeasure for the seen-bias diagnosis (E3..E12):
        # after locking Ws, the trainable visual projector was the only remaining
        # fitted block. With wv_init="identity" the projector starts from the
        # pre-aligned geometry instead of a random projection:
        # - single path: Wv_single <- I (requires input_dim == out_dim);
        # - pyramid path: level_proj <- I and Wv <- [I;I;I]/sqrt(3), so the
        #   pooled vector is the (scaled) mean of the levels at init.
        # wv_anchor>0: trainer adds ||Wv - Wv_init||_F^2 to the loss so Wv can
        # only deviate with a cost. freeze_wv: hard-freeze at init.
        self.wv_init = str(getattr(aggregator_cfg, "wv_init", "default")).lower()
        if self.wv_init not in ("default", "identity"):
            raise ValueError("model.aggregator.wv_init must be 'default' or 'identity'.")
        self.wv_anchor = float(getattr(aggregator_cfg, "wv_anchor", 0.0))
        self.freeze_wv = bool(getattr(aggregator_cfg, "freeze_wv", False))
        if (self.wv_anchor > 0 or self.freeze_wv) and self.wv_init != "identity":
            raise ValueError(
                "wv_anchor/freeze_wv require model.aggregator.wv_init='identity'."
            )
        if self.wv_init == "identity":
            with torch.no_grad():
                if self.single_dim == self.out_dim:
                    nn.init.eye_(self.Wv_single.weight)
                if self.pyramid:
                    if any(dim != self.level_proj_dim for dim in self.level_dims) \
                            or self.level_proj_dim != self.out_dim:
                        raise ValueError(
                            "Pyramid identity init requires level_dims == level_dim == out_dim, "
                            f"got {self.level_dims} / {self.level_proj_dim} / {self.out_dim}."
                        )
                    for proj in self.level_proj:
                        nn.init.eye_(proj.weight)
                    scale = 1.0 / math.sqrt(len(self.level_proj))
                    eye = torch.eye(self.out_dim)
                    self.Wv.weight.copy_(
                        torch.cat([eye] * len(self.level_proj), dim=1) * scale
                    )
                elif self.level_dims == [self.out_dim]:
                    nn.init.eye_(self.Wv.weight)
            if self.freeze_wv:
                for proj in self.level_proj or []:
                    proj.requires_grad_(False)
                self.Wv.requires_grad_(False)
                if self.single_dim == self.out_dim:
                    self.Wv_single.requires_grad_(False)
        # Anchor reference weights for the trainer's wv_anchor penalty
        # (None when disabled).
        self._wv_anchor_state = None
        if self.wv_anchor > 0:
            with torch.no_grad():
                parts = [proj.weight.detach().clone() for proj in (self.level_proj or [])]
                parts.append(self.Wv.weight.detach().clone())
                if self.single_dim == self.out_dim:
                    parts.append(self.Wv_single.weight.detach().clone())
            self._wv_anchor_state = parts

    @staticmethod
    def gap(feature: torch.Tensor) -> torch.Tensor:
        if feature.dim() == 4:
            return feature.mean(dim=(2, 3))
        if feature.dim() == 3:
            return feature.mean(dim=1)
        if feature.dim() == 2:
            return feature
        raise ValueError(
            "Visual feature must have shape [B,D], [B,N,D], or [B,D,H,W]."
        )

    def forward(self, features: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        """Return normalized image representation v_tilde with shape [B, d]."""
        if isinstance(features, dict):
            # v10 anchored path: {"tokens": [z_l, z_m, z_h], "cls": [B, 512]}.
            if self.mode == "cls_only":
                return F.normalize(features["cls"], p=2, dim=-1)
            if self.mode != "anchored":
                raise ValueError(
                    "Dict input (cls + tokens) requires model.aggregator.mode "
                    f"'anchored' or 'cls_only', got '{self.mode}'."
                )
            tokens = features["tokens"]
            cls = features["cls"]
            if len(tokens) != len(self.level_dims):
                raise ValueError(
                    f"Anchored MFA expects {len(self.level_dims)} token levels, "
                    f"got {len(tokens)}."
                )
            pooled = [self.gap(t) for t in tokens]
            visual = torch.cat(pooled, dim=-1)
            if visual.shape[-1] != self.W_multi.in_features:
                raise ValueError(
                    f"Expected visual dimension {self.W_multi.in_features}, "
                    f"got {visual.shape[-1]}."
                )
            if visual.shape[0] != cls.shape[0] or cls.shape[-1] != self.out_dim:
                raise ValueError(
                    f"cls anchor shape {tuple(cls.shape)} incompatible with "
                    f"tokens {tuple(visual.shape)} / out_dim {self.out_dim}."
                )
            # Epoch-0 exact anchor: W_multi == 0 => v_tilde == normalize(cls).
            return F.normalize(cls + self.W_multi(visual), p=2, dim=-1)
        if isinstance(features, Sequence) and not torch.is_tensor(features):
            if len(features) == 0:
                raise ValueError("At least one visual feature level is required.")
            pooled = [self.gap(feature) for feature in features]
            if self.pyramid:
                if len(pooled) != len(self.level_proj):
                    raise ValueError(
                        "Pyramid aggregator expects "
                        f"{len(self.level_proj)} levels, got {len(pooled)}."
                    )
                projected_levels = [
                    proj(level) for proj, level in zip(self.level_proj, pooled)
                ]
                visual = torch.cat(projected_levels, dim=-1)
            else:
                visual = torch.cat(pooled, dim=-1)
            projector = self.Wv
        else:
            visual = self.gap(features)
            projector = self.Wv_single

        if visual.shape[-1] != projector.in_features:
            raise ValueError(
                f"Expected visual dimension {projector.in_features}, "
                f"got {visual.shape[-1]}."
            )

        projected = projector(visual)
        return F.normalize(projected, p=2, dim=-1)
# if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('🚀🚀🚀🚀🚀 ', device)
    # C = type('Config', (object,), {})
    # c = C(); c.model = C(); c.model.aggregator = C()
    # setattr(c.model.aggregator, 'dim_l', 768)
    # setattr(c.model.aggregator, 'dim_m', 768)
    # setattr(c.model.aggregator, 'dim_h', 768)
    # setattr(c.model.aggregator, 'out_dim', 512)
    # setattr(c.model.aggregator, 'input_dim', 2048)

    # agg = MultiFeatureAggregator(c).to(device)

    # # 1. ViT-style: list of [B, N, D]
    # B = 2
    # vit_feats = [torch.randn(B, 197, 768).to(device) for _ in range(3)]
    # out = agg(vit_feats)
    # assert out.shape == (B, 512)
    # ones = torch.ones(B, device=device)
    # assert torch.allclose(out.norm(p=2, dim=-1), ones)
    # print(f"[ViT] v_tilde {out.shape} normalized: OK")

    # # 2. ResNet-style: list of [B, C, H, W]
    # c2 = C(); c2.model = C(); c2.model.aggregator = C()
    # setattr(c2.model.aggregator, 'level_dims', [256, 512, 1024])
    # setattr(c2.model.aggregator, 'out_dim', 512)
    # agg2 = MultiFeatureAggregator(c2).to(device)
    # res_feats = [
    #     torch.randn(B, 256, 56, 56).to(device),
    #     torch.randn(B, 512, 28, 28).to(device),
    #     torch.randn(B, 1024, 14, 14).to(device),
    # ]
    # out2 = agg2(res_feats)
    # assert out2.shape == (B, 512)
    # assert torch.allclose(out2.norm(p=2, dim=-1), ones)
    # print(f"[ResNet] v_tilde {out2.shape} normalized: OK")

    # # 3. CLIP pooled: single [B, D]
    # clip_feat = torch.randn(B, 2048).to(device)
    # out3 = agg(clip_feat)
    # assert out3.shape == (B, 512)
    # assert torch.allclose(out3.norm(p=2, dim=-1), ones)
    # print(f"[CLIP pooled] v_tilde {out3.shape} normalized: OK")

    # # 4. gap unit checks
    # assert agg.gap(torch.randn(B, 768).to(device)).shape == (B, 768)
    # assert agg.gap(torch.randn(B, 197, 768).to(device)).shape == (B, 768)
    # assert agg.gap(torch.randn(B, 256, 56, 56).to(device)).shape == (B, 256)
    # print("[gap] 2D/3D/4D shapes OK")

    # # 5. Single 3D/4D tensor path
    # assert agg(torch.randn(B, 197, 2048).to(device)).shape == (B, 512)
    # assert agg(torch.randn(B, 2048, 7, 7).to(device)).shape == (B, 512)
    # print("[single 3D/4D] shapes OK")

    # print("All tests passed!")