import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiFeatureAggregator(nn.Module):
    """v14 anchored Multi-level Feature Aggregator (đường duy nhất).

    Visual encoder trả {"tokens": [z_l, z_m, z_h], "cls": [B, D]} từ MỘT
    forward, và:
      v_global = normalize(cls + W_multi[GAP(z_l); GAP(z_m); GAP(z_h)])
    W_multi zero-init (LP-FT anchor) => epoch-0 v_global == normalize(cls).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        aggregator_cfg = cfg.model.aggregator
        self.out_dim = getattr(aggregator_cfg, "out_dim", 512)
        self.level_dims = list(aggregator_cfg.level_dims)
        self.W_multi = nn.Linear(sum(self.level_dims), self.out_dim, bias=True)
        nn.init.zeros_(self.W_multi.weight)
        nn.init.zeros_(self.W_multi.bias)

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

    def forward(self, features: dict) -> torch.Tensor:
        """Return normalized image representation v_global with shape [B, d]."""
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
        # Epoch-0 exact anchor: W_multi == 0 => v_global == normalize(cls).
        return F.normalize(cls + self.W_multi(visual), p=2, dim=-1)
