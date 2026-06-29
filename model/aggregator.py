from collections.abc import Sequence

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

        self.Wv = nn.Linear(sum(self.level_dims), self.out_dim, bias=False)
        self.Wv_single = nn.Linear(self.single_dim, self.out_dim, bias=False)

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
        if isinstance(features, Sequence) and not torch.is_tensor(features):
            if len(features) == 0:
                raise ValueError("At least one visual feature level is required.")
            pooled = [self.gap(feature) for feature in features]
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