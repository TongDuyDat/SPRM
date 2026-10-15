"""Asymmetric Loss (ASL) for multi-label learning (Ridnik et al., ICCV 2021).

Moved here verbatim from ``train_coco_plan.py`` (where it was defined) so that
the multi-label path no longer imports its loss from a training script. The
formula is unchanged: this is the exact implementation that produced the
multi-label numbers, so any edit here invalidates them.

Registered as ``model.loss.type: "AsymmetricLoss"``.
"""

import torch
import torch.nn as nn

from tools.registry import LOSSES


@LOSSES.register("AsymmetricLoss")
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) cho multi-label learning (Ridnik et al., ICCV 2021).

    Args:
        gamma_neg: focusing parameter for the negative (unlabeled) class term.
        gamma_pos: focusing parameter for the positive term (0 = plain BCE).
        clip: shifts the negative probability by this amount to drop the
            hardest negatives (``xs_neg + clip``, clamped at 1).

    Note on scaling: the training scripts call this as
    ``criterion(scores / tau, targets)`` with ``tau=0.07``. ASL expects plain
    logits (its own sigmoid is applied inside), so dividing by a small tau
    saturates the sigmoid and degenerates the negative term. That behaviour is
    preserved here because it is what produced the reported numbers; see the
    unification plan before changing it.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1.0 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss *= one_sided_w
        return -loss.sum() / max(1, x.size(0))


def build_asymmetric_loss(cfg):
    """Build from ``cfg.model.loss`` (the ``loss.type`` dispatch target)."""
    loss_cfg = cfg.model.loss
    return AsymmetricLoss(
        gamma_neg=float(getattr(loss_cfg, "asl_gamma_neg", 4.0)),
        gamma_pos=float(getattr(loss_cfg, "asl_gamma_pos", 0.0)),
        clip=float(getattr(loss_cfg, "asl_clip", 0.05)),
    )
