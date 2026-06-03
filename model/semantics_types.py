from dataclasses import dataclass
import torch

@dataclass
class SemanticPack:
    # Standardized semantic tensors passed between modules to make loss heads modular
    phi_t: torch.Tensor | None          # [C, D_text] raw text encoder output or cached embeddings
    p_sem: torch.Tensor | None          # [C, D] normalized semantic prototypes
    p_sem_unnorm: torch.Tensor | None   # [C, D] before L2 normalization
