import torch
import torch.nn as nn
import torch.nn.functional as F

class HSPRNLoss(nn.Module):
    """
    4.8 Training Objective
    Computes the total loss for the HSPRN framework.
    """
    def __init__(self, cfg):
        super(HSPRNLoss, self).__init__()
        self.cfg = cfg
        
        self.lambda1 = getattr(cfg.model.loss, 'lambda1', 1.0)
        self.lambda2 = getattr(cfg.model.loss, 'lambda2', 0.1)
        self.lambda3 = getattr(cfg.model.loss, 'lambda3', 0.1)
        self.lambda4 = getattr(cfg.model.loss, 'lambda4', 0.1)
        
        self.tau = getattr(cfg.model.loss, 'temperature', 0.1)

    def forward(self, s_c, y, v_tilde, phi_t, p_sem, p_ref):
        """
        Args:
            s_c: Compatibility scores (B, C)
            y: Ground-truth class labels (B,)
            v_tilde: Aggregated visual features (B, D)
            phi_t: Encoded semantic descriptions (C, D_text)
            p_sem: Original semantic prototypes (C, D)
            p_ref: Refined semantic prototypes (B, C, D)
        Returns:
            total_loss, loss_dict
        """
        B, C = s_c.shape
        
        # 1. Classification Loss (Eq. 32)
        # Lcls = -log(exp(s_y / tau) / sum(exp(s_c / tau)))
        # This is equivalent to CrossEntropyLoss with logits = s_c / tau
        l_cls = F.cross_entropy(s_c / self.tau, y)
        
        # 2. Semantic Consistency Loss (Eq. 33)
        # Lsem = 1 - cos(p_sem_c, phi(t_c))
        # Since p_sem = Ws(phi_t) and is L2 normalized, 
        # we compute cosine similarity between unnormalized phi_t and normalized p_sem.
        # Assuming phi_t and p_sem have the same dimension (e.g. after PriSematicConstructor projection):
        if phi_t.shape[-1] == p_sem.shape[-1]:
            phi_t_norm = F.normalize(phi_t, p=2, dim=-1)
            # Both are (C, D). Cosine sim for each class:
            sim_sem = (p_sem * phi_t_norm).sum(dim=-1)
            l_sem = (1.0 - sim_sem).mean()
        else:
            l_sem = torch.tensor(0.0).to(s_c.device)
            
        # 3. Visual-Semantic Alignment Loss (Eq. 34)
        # Lalign = 1 - cos(\tilde{v}, \tilde{p}^{sem}_y)
        # Extract the semantic prototype corresponding to ground-truth label y for each sample in the batch
        p_sem_y = p_sem[y] # shape (B, D)
        # \tilde{v} is (B, D), p_sem_y is (B, D). Both are L2-normalized.
        if v_tilde.shape[-1] == p_sem_y.shape[-1]:
            sim_align = (v_tilde * p_sem_y).sum(dim=-1) # shape (B,)
            l_align = (1.0 - sim_align).mean()
        else:
            l_align = torch.tensor(0.0).to(s_c.device)
            
        # 4. Prototype Preservation Loss (Eq. 35)
        # Lpres = || \tilde{p}^{ref}_y - \tilde{p}^{sem}_y ||_2^2
        # Extract refined prototype for ground-truth class y
        # p_ref is (B, C, D). We want p_ref[i, y[i], :] for each sample i.
        batch_indices = torch.arange(B).to(y.device)
        p_ref_y = p_ref[batch_indices, y] # shape (B, D)
        
        # Mean squared error (L2 distance squared)
        l_pres = F.mse_loss(p_ref_y, p_sem_y)
        
        # Overall Loss (Eq. 36)
        total_loss = (
            self.lambda1 * l_cls + 
            self.lambda2 * l_sem + 
            self.lambda3 * l_align + 
            self.lambda4 * l_pres
        )
        
        loss_dict = {
            'loss_cls': l_cls.item(),
            'loss_sem': l_sem.item(),
            'loss_align': l_align.item(),
            'loss_pres': l_pres.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
