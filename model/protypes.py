import torch
import torch.nn as nn
import torch.nn.functional as F

class SematicBank(nn.Module):
    """
    4.2.3. Semantic Encoding & 4.2.4. Output
    Implements the semantic prototype bank.
    """
    def __init__(self, cfg):
        super(SematicBank, self).__init__()
        self.cfg = cfg
        # Ws is a learnable projection matrix
        # Projects from encoded text feature dim to shared semantic space dim
        text_dim = getattr(cfg.model.semantic, 'text_dim', 768)
        shared_dim = getattr(cfg.model.semantic, 'shared_dim', 512)
        
        self.Ws = nn.Linear(text_dim, shared_dim, bias=False)

    def forward(self, phi_t):
        """
        Args:
            phi_t: Encoded semantic features phi(tc) from text encoder, shape (..., text_dim)
        Returns:
            Psem: Normalized semantic prototype bank, shape (..., shared_dim)
        """
        # Eq 6: psem_c = Ws phi(tc)
        psem_c = self.Ws(phi_t)
        
        # Eq 7: ~psem_c = psem_c / ||psem_c||_2
        psem_normalized = F.normalize(psem_c, p=2, dim=-1)
        
        # Eq 8: Output is the semantic prototype bank Psem
        return psem_normalized

class InstancePrototypeRefiner(nn.Module):
    """
    4.5 Instance-aware Prototype Refiner
    Refines semantic prototypes using visual features.
    """
    def __init__(self, cfg):
        super(InstancePrototypeRefiner, self).__init__()
        self.cfg = cfg
        
        # shared_dim is the dimension of the prototype (e.g. 512)
        self.shared_dim = getattr(cfg.model.semantic, 'shared_dim', 512)
        
        # Visual feature dimension might be different (e.g., zh from ViT is 768, or ResNet is 1024)
        # We assume the visual encoder highest level output has this dimension
        self.vis_dim = getattr(cfg.model.aggregator, 'dim_h', 768) 
        
        # Matrix Wr for the gating vector (Eq. 19).
        # Input to Wr is concat of [v, p, v \odot p].
        # However, v and p might have different dimensions, so we project v first.
        self.U = nn.Linear(self.vis_dim, self.shared_dim, bias=False) # Eq. 20
        
        # After projecting v to q, both q and p have dimension shared_dim.
        # So concat [q; p; q \odot p] has size 3 * shared_dim.
        self.Wr = nn.Linear(3 * self.shared_dim, self.shared_dim)

    def forward(self, p_sem, img_feat):
        """
        Args:
            p_sem: \tilde{p}^{sem}_c, shape (..., shared_dim)
            img_feat: Visual features Z from visual encoder. Usually Z = (zl, zm, zh).
                      We'll extract the highest level feature (zh) or pool it.
        Returns:
            p_hat: Refined unnormalized prototypes, shape (..., shared_dim)
        """
        # Get highest-level visual feature zh
        if isinstance(img_feat, tuple) or isinstance(img_feat, list):
            zh = img_feat[-1]
        else:
            zh = img_feat
            
        # GAP if zh is spatial [B, N, C] or [B, C, H, W]
        if zh.dim() == 3:
            zh = zh.mean(dim=1)
        elif zh.dim() == 4:
            zh = zh.mean(dim=[2, 3])
            
        # Eq 20: q = U \tilde{v} (Here we use zh as the visual feature for refinement)
        q = self.U(zh) # shape: (B, shared_dim)
        
        # To compute gating vector for all classes, we need to broadcast.
        # q is (B, D). p_sem is (C, D).
        # We unsqueeze to make q (B, 1, D) and p_sem (1, C, D)
        B = q.shape[0]
        if p_sem.dim() == 2:
            C = p_sem.shape[0]
            q_expand = q.unsqueeze(1).expand(B, C, -1)
            p_expand = p_sem.unsqueeze(0).expand(B, C, -1)
        else:
            # If p_sem is already broadcasted (B, C, D)
            q_expand = q.unsqueeze(1).expand(-1, p_sem.shape[1], -1)
            p_expand = p_sem
            
        # Eq 19: g_c = \sigma(W_r [\tilde{v}; \tilde{p}^{sem}_c; \tilde{v} \odot \tilde{p}^{sem}_c])
        # We use q instead of \tilde{v} since it is already projected to the right dimension.
        concat_feat = torch.cat([
            q_expand, 
            p_expand, 
            q_expand * p_expand
        ], dim=-1) # shape: (B, C, 3*shared_dim)
        
        gc = torch.sigmoid(self.Wr(concat_feat)) # shape: (B, C, shared_dim)
        
        # Eq 21: \hat{p}_c = \tilde{p}^{sem}_c + g_c \odot q
        p_hat = p_expand + gc * q_expand
        
        return p_hat


class RefinedPrototypeBank(nn.Module):
    """
    4.5.5 Output (Refined Prototype Bank)
    Normalizes the refined prototypes.
    """
    def __init__(self, cfg):
        super(RefinedPrototypeBank, self).__init__()
        
    def forward(self, p_hat):
        """
        Args:
            p_hat: Refined prototypes \hat{p}_c
        Returns:
            p_ref: \tilde{p}^{ref}_c (Normalized), shape (B, C, D)
        """
        # Eq 22: \tilde{p}^{ref}_c = \hat{p}_c / ||\hat{p}_c||_2
        p_ref = F.normalize(p_hat, p=2, dim=-1)
        return p_ref
