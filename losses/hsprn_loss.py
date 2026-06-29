import torch
import torch.nn as nn
import torch.nn.functional as F

class HSPRNLoss(nn.Module):
    """
    4.8 Training Objective
    Computes Eq. (32), (34)-(36) from
    docs/Hierarchical_Semantic_Prototype_Refinement_Model__HSPRM_.md.
    """
    def __init__(self, cfg):
        super(HSPRNLoss, self).__init__()
        self.cfg = cfg

        loss_cfg = cfg.model.loss
        required = ("lambda1", "lambda2", "lambda3", "temperature")
        missing = [name for name in required if not hasattr(loss_cfg, name)]
        if missing:
            raise ValueError(
                "HSPRN loss hyperparameters must be set in config: "
                + ", ".join(missing)
            )
        self.lambda1 = loss_cfg.lambda1
        self.lambda2 = loss_cfg.lambda2
        self.lambda3 = loss_cfg.lambda3
        self.tau = loss_cfg.temperature

    def forward(self, s_c, y, v_tilde=None, p_sem=None, p_ref=None):
        """
        Args:
            s_c: Compatibility scores (B, C)
            y: Ground-truth class labels (B,)
            v_tilde: Aggregated visual features (B, D)
            p_sem: Normalized semantic prototypes (C, D), i.e. tilde p_sem
            p_ref: Refined semantic prototypes (B, C, D)
        Returns:
            total_loss, loss_dict
        """
        B = s_c.shape[0]
        required = {
            "v_tilde": v_tilde,
            "p_sem": p_sem,
            "p_ref": p_ref,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Full HSPRN loss requires intermediate tensors; missing: "
                + ", ".join(missing)
            )

        # 1) Classification (Eq. 32)
        l_cls = F.cross_entropy(s_c / self.tau, y)

        # 2) Visual-Semantic alignment (Eq. 34)
        if v_tilde.shape[-1] != p_sem.shape[-1]:
            raise ValueError(
                "Eq. (34) requires v_tilde and tilde p_sem_y to share the same dimension, "
                f"got {v_tilde.shape[-1]} and {p_sem.shape[-1]}."
            )
        p_sem_y = p_sem[y]
        l_align = (1.0 - F.cosine_similarity(v_tilde, p_sem_y, dim=-1)).sum()

        # 3) Prototype preservation (Eq. 35)
        if p_ref.shape[-1] != p_sem.shape[-1]:
            raise ValueError(
                "Eq. (35) requires tilde p_ref_y and tilde p_sem_y to share the same "
                f"dimension, got {p_ref.shape[-1]} and {p_sem.shape[-1]}."
            )
        batch_indices = torch.arange(B, device=y.device)
        p_ref_y = p_ref[batch_indices, y]
        p_sem_y = p_sem[y]
        l_pres = ((p_ref_y - p_sem_y) ** 2).sum(dim=-1).sum()

        total_loss = (
            self.lambda1 * l_cls +
            self.lambda2 * l_align +
            self.lambda3 * l_pres
        )

        loss_dict = {
            'loss_cls': float(l_cls.detach().cpu()),
            'loss_align': float(l_align.detach().cpu()),
            'loss_pres': float(l_pres.detach().cpu()),
            'total_loss': float(total_loss.detach().cpu()),
        }
        return total_loss, loss_dict
