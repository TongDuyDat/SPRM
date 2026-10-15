import torch
import torch.nn as nn
import torch.nn.functional as F


class SPRNLoss(nn.Module):
    """Tri-Loss: L_task + L_visual + L_attr + L_cls + L_rel.

    The trainer overwrites the five lambda_* attributes at every phase transition
    (the attribute names are load-bearing); they start at 0.
    """

    def __init__(self, cfg):
        super().__init__()
        self.tau = float(getattr(cfg.model.loss, "temperature", 0.07))
        self.lambda_task = 0.0
        self.lambda_visual = 0.0
        self.lambda_attr = 0.0
        self.lambda_cls = 0.0
        self.lambda_rel = 0.0

    def forward(self, s_c, y, v_tilde, cls_anchor, scores_all, phi_all, attr_pred,
                gt_attr, p_sem_all_rel):
        """Returns total_loss, loss_dict."""
        # 1) Task: full-bank zero-shot soft distillation from the frozen CLIP CLS.
        if scores_all.shape[-1] != phi_all.shape[0]:
            raise ValueError(
                f"Tri-Loss L_task shape mismatch: scores_all has {scores_all.shape[-1]} "
                f"columns, phi_all has {phi_all.shape[0]} classes."
            )
        with torch.no_grad():
            phi_n = F.normalize(phi_all, dim=-1)
            q = F.softmax((cls_anchor @ phi_n.T) / self.tau, dim=-1)
        log_p = F.log_softmax(scores_all / self.tau, dim=-1)
        l_task = F.kl_div(log_p, q, reduction="batchmean")

        # Optional seen-only hard CE.
        l_cls = None
        if self.lambda_cls > 0:
            l_cls = F.cross_entropy(s_c / self.tau, y)

        # 2) Visual anchor: cosine retention to the CLIP CLS.
        l_visual = (1.0 - F.cosine_similarity(v_tilde, cls_anchor, dim=-1)).mean()

        # 3) Attribute grounding of the slot attention peaks.
        l_attr = None
        if self.lambda_attr > 0:
            l_attr = (1.0 - F.cosine_similarity(attr_pred, F.normalize(gt_attr, dim=-1), dim=-1)).mean()

        # 4) Relational KD for Ws on the full prototype bank.
        l_rel = None
        if self.lambda_rel > 0:
            with torch.no_grad():
                phi_n = F.normalize(phi_all, dim=-1)
                p0 = F.softmax(-torch.cdist(phi_n, phi_n), dim=-1)
            p1 = F.log_softmax(-torch.cdist(p_sem_all_rel, p_sem_all_rel), dim=-1)
            l_rel = F.kl_div(p1, p0, reduction="batchmean")

        total_loss = self.lambda_task * l_task + self.lambda_visual * l_visual
        if l_attr is not None:
            total_loss = total_loss + self.lambda_attr * l_attr
        if l_cls is not None:
            total_loss = total_loss + self.lambda_cls * l_cls
        if l_rel is not None:
            total_loss = total_loss + self.lambda_rel * l_rel

        loss_dict = {
            "loss_task": float((self.lambda_task * l_task).detach().cpu()),
            "loss_visual": float((self.lambda_visual * l_visual).detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "cos_v_cls": float(F.cosine_similarity(v_tilde, cls_anchor, dim=-1).mean().detach().cpu()),
        }
        if l_attr is not None:
            loss_dict["loss_attr"] = float((self.lambda_attr * l_attr).detach().cpu())
        if l_cls is not None:
            loss_dict["loss_cls"] = float((self.lambda_cls * l_cls).detach().cpu())
        if l_rel is not None:
            loss_dict["loss_rel"] = float((self.lambda_rel * l_rel).detach().cpu())
        return total_loss, loss_dict
