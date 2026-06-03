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

    def as_tuple(self):
        return (self.scores, self.v_embed, self.p_sem, self.p_ref, self.phi_t, self.p_sem_unnorm)
