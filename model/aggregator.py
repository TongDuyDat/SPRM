import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiFeatureAggregator(nn.Module):
    """
    4.4 Multi-level Feature Aggregator (Module J)
    Combines multi-level visual features into a unified image representation.
    """
    def __init__(self, cfg):
        super(MultiFeatureAggregator, self).__init__()
        self.cfg = cfg
        
        # Dimensions of zl, zm, zh. 
        # For ViT Base, usually they are all 768.
        # For ResNet, they might be 256, 512, 1024 respectively.
        self.dim_l = getattr(cfg.model.aggregator, 'dim_l', 768)
        self.dim_m = getattr(cfg.model.aggregator, 'dim_m', 768)
        self.dim_h = getattr(cfg.model.aggregator, 'dim_h', 768)
        
        # Final output dimension d
        self.d = getattr(cfg.model.aggregator, 'out_dim', 512)
        
        # Since the concatenated feature size could be dim_l + dim_m + dim_h
        concat_dim = self.dim_l + self.dim_m + self.dim_h
        
        # Projection matrix W_v (Eq. 15)
        self.Wv = nn.Linear(concat_dim, self.d, bias=False)

    def forward(self, multi_level_features):
        """
        Args:
            multi_level_features: Tuple of (zl, zm, zh)
                Each feature can be a patch token sequence [B, N, C] or spatial map [B, C, H, W]
                or just a pooled vector [B, C].
        Returns:
            v_tilde: Aggregated and normalized visual representation, shape (B, d)
        """
        zl, zm, zh = multi_level_features
        
        # Helper function for Global Average Pooling
        def gap(x):
            # If x is [B, N, C] (e.g. ViT patch tokens), mean over N
            if x.dim() == 3:
                return x.mean(dim=1)
            # If x is [B, C, H, W] (e.g. ResNet feature maps), mean over H and W
            elif x.dim() == 4:
                return x.mean(dim=[2, 3])
            # If already pooled [B, C], just return
            return x

        # Eq. 14: GAP(zl), GAP(zm), GAP(zh)
        pool_l = gap(zl)
        pool_m = gap(zm)
        pool_h = gap(zh)
        
        # Eq. 15: v = Wv[GAP(zl); GAP(zm); GAP(zh)]
        concat_v = torch.cat([pool_l, pool_m, pool_h], dim=-1)
        v = self.Wv(concat_v)
        
        # Eq. 16: \tilde{v} = v / ||v||_2
        v_tilde = F.normalize(v, p=2, dim=-1)
        
        return v_tilde
