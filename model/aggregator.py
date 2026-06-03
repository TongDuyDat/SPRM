import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Union, Optional, List

class MultiFeatureAggregator(nn.Module):
    """
    4.4 Multi-level Feature Aggregator (Module J)
    Combines multi-level visual features into a unified image representation.

    Flexible inputs:
    - Sequence of feature maps/tokens: [zl, zm, zh, ...] where each item may be
      [B, N, C], [B, C, H, W], or [B, C]. Any number of levels is supported.
    - Single 2D tensor [B, D] (pre-extracted visual features).

    Implementation follows Eqs. (14)–(16):
      v  = Wv [GAP(z1); GAP(z2); ...]
      v~ = v / ||v||_2

    The projection Wv is created lazily with the appropriate input dimension the
    first time forward() is called. This makes the module robust to changing the
    selected visual levels from config without touching the code.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d = getattr(cfg.model.aggregator, 'out_dim', 512)
        # Lazy projector built on first use (in_features determined from input)
        self.Wv_dyn: Optional[nn.Linear] = None

    @staticmethod
    def _gap(x: torch.Tensor) -> torch.Tensor:
        # If x is [B, N, C] (e.g. ViT tokens), mean over N
        if x.dim() == 3:
            return x.mean(dim=1)
        # If x is [B, C, H, W] (e.g. CNN feature maps), mean over H and W
        if x.dim() == 4:
            return x.mean(dim=[2, 3])
        # If already pooled [B, C], just return
        return x

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        in_dim = z.shape[-1]
        if self.Wv_dyn is None or self.Wv_dyn.in_features != in_dim:
            self.Wv_dyn = nn.Linear(in_dim, self.d, bias=False)
            self.Wv_dyn.to(z.device, dtype=z.dtype)
        v = self.Wv_dyn(z)
        return F.normalize(v, p=2, dim=-1)

    def forward(self, feats: Union[torch.Tensor, Sequence[torch.Tensor]]):
        """
        Args:
            feats: list/tuple of feature tensors from multiple levels, or a single
                   2D tensor [B, D].
        Returns:
            v_tilde: [B, d] aggregated and L2-normalized visual vector.
        """
        if isinstance(feats, (list, tuple)):
            pooled: List[torch.Tensor] = [self._gap(x) for x in feats]
            z = torch.cat(pooled, dim=-1)
            return self._project(z)
        # Single tensor case
        z = feats
        if z.dim() != 2:
            z = self._gap(z)
        return self._project(z)
