import torch
import torch.nn as nn
import torch.nn.functional as F

class SematicMatchingHead(nn.Module):
    """
    4.6 Semantic Matching Head (Module F)
    Matches visual features with refined prototypes using cosine similarity.
    """
    def __init__(self, cfg):
        super(SematicMatchingHead, self).__init__()
        self.cfg = cfg
        
        self.vis_dim = getattr(cfg.model.aggregator, 'out_dim', 512)
        self.shared_dim = getattr(cfg.model.semantic, 'shared_dim', 512)
        
        # If the visual output dimension (v_tilde) does not match the prototype dimension,
        # add a projection layer. Though usually they are designed to match (e.g., both 512).
        if self.vis_dim != self.shared_dim:
            self.proj = nn.Linear(self.vis_dim, self.shared_dim, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, v_tilde, p_ref):
        """
        Args:
            v_tilde: Aggregated visual features, shape (B, vis_dim)
            p_ref: Refined semantic prototype bank, shape (B, C, shared_dim) 
                   (or (C, shared_dim) if not instance-aware, but here it is (B, C, shared_dim))
        Returns:
            s_c: Compatibility scores (cosine similarity), shape (B, C)
        """
        # Align dimension if necessary
        v = self.proj(v_tilde)
        
        # Ensure v is L2 normalized if we projected it, although v_tilde is already normalized
        if self.vis_dim != self.shared_dim:
            v = F.normalize(v, p=2, dim=-1)
            
        # Eq 26: s_c = cos(\tilde{v}, \tilde{p}^{ref}_c)
        # Since both v and p_ref are L2-normalized, cosine similarity is just the dot product.
        
        # v shape: (B, D) -> unsqueeze to (B, 1, D)
        v = v.unsqueeze(1)
        
        # p_ref shape: (B, C, D)
        # Dot product over the D dimension
        s_c = (v * p_ref).sum(dim=-1) # shape: (B, C)
        
        return s_c
