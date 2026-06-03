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

    def forward(self, s_c, y, v_tilde=None, phi_t=None, p_sem=None, p_ref=None, p_sem_unnorm=None):
        """
        Args:
            s_c: Compatibility scores (B, C)
            y: Ground-truth class labels (B,)
            v_tilde: Aggregated visual features (B, D)
            phi_t: Encoded semantic descriptions (C, D_text) or cached embeddings
            p_sem: Normalized semantic prototypes (C, D)
            p_ref: Refined semantic prototypes (B, C, D)
            p_sem_unnorm: Unnormalized semantic prototypes (C, D)
        Returns:
            total_loss, loss_dict
        Notes:
            - Any missing component gracefully disables its corresponding term.
            - Classification loss is always computed from s_c and y.
        """
        B, C = s_c.shape

        # 1) Classification (Eq. 32)
        l_cls = F.cross_entropy(s_c / self.tau, y)

        # Defaults
        zero = torch.tensor(0.0, device=s_c.device, dtype=s_c.dtype)
        l_sem = zero
        l_align = zero
        l_pres = zero

        # 2) Semantic consistency (Eq. 33)
        # Prefer comparing in the same space: cos(normalize(p_sem_unnorm), p_sem) if both present.
        if p_sem is not None:
            if p_sem_unnorm is not None:
                p_sem_from_raw = F.normalize(p_sem_unnorm, p=2, dim=-1)
                sim_sem = (p_sem_from_raw * p_sem).sum(dim=-1)
                l_sem = (1.0 - sim_sem).mean()
            elif phi_t is not None and phi_t.shape[-1] == p_sem.shape[-1]:
                phi_t_norm = F.normalize(phi_t, p=2, dim=-1)
                sim_sem = (p_sem * phi_t_norm).sum(dim=-1)
                l_sem = (1.0 - sim_sem).mean()

        # 3) Visual–Semantic alignment (Eq. 34)
        if v_tilde is not None and p_sem is not None and v_tilde.shape[-1] == p_sem.shape[-1]:
            p_sem_y = p_sem[y]
            sim_align = (v_tilde * p_sem_y).sum(dim=-1)
            l_align = (1.0 - sim_align).mean()

        # 4) Prototype preservation (Eq. 35)
        if p_ref is not None and p_sem is not None and p_ref.shape[-1] == p_sem.shape[-1]:
            batch_indices = torch.arange(B, device=y.device)
            p_ref_y = p_ref[batch_indices, y]
            p_sem_y = p_sem[y]
            l_pres = F.mse_loss(p_ref_y, p_sem_y)

        total_loss = (
            self.lambda1 * l_cls +
            self.lambda2 * l_sem +
            self.lambda3 * l_align +
            self.lambda4 * l_pres
        )

        loss_dict = {
            'loss_cls': float(l_cls.detach().cpu()),
            'loss_sem': float(l_sem.detach().cpu()),
            'loss_align': float(l_align.detach().cpu()),
            'loss_pres': float(l_pres.detach().cpu()),
            'total_loss': float(total_loss.detach().cpu()),
        }
        return total_loss, loss_dict
