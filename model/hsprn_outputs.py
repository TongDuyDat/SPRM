from dataclasses import dataclass
import torch

@dataclass
class HSPRNOutput:
    # Required for classification & evaluation
    scores: torch.Tensor          # [B, C]
    # Optional but standardized for losses/analysis
    v_embed: torch.Tensor | None = None        # [B, D] aggregated visual v~
    p_sem: torch.Tensor | None = None          # [C, D] normalized semantic prototypes
    p_ref: torch.Tensor | None = None          # [B, C, D] refined normalized prototypes
    phi_t: torch.Tensor | None = None          # [C, D_text] raw semantic encoder outputs (or cached semantic input)
    p_sem_unnorm: torch.Tensor | None = None   # [C, D] before L2 norm (if available)
    scores_sem: torch.Tensor | None = None     # [B, C] scores against UNREFINED semantic
                                               # prototypes (unrefined scoring path)

    # v10: per-image attribute presence predictions [B, n_attr] from the
    # cross-attention IPR attribute branch (None on the legacy path).
    attr_pred: torch.Tensor | None = None
    # v10: deepest patch-token map [B, N, 768] from extract_levels_anchored
    # (None on the legacy path) — needed when the trainer re-runs the
    # cross-attention IPR on the full-class bank for L_pres.
    tokens: torch.Tensor | None = None
    # v11: frozen zero-shot anchor normalize(features["cls"]) from the
    # anchored MFA path — used by L_fm (cosine retention) and as the
    # teacher visual embedding for L_zs (zero-shot posterior distillation).
    cls_anchor: torch.Tensor | None = None

    def as_tuple(self):
        return (self.scores, self.v_embed, self.p_sem, self.p_ref, self.phi_t, self.p_sem_unnorm)
