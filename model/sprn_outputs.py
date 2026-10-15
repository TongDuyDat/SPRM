from dataclasses import dataclass
import torch


@dataclass
class SPRNOutput:
    scores: torch.Tensor        # [B, C] compatibility scores
    v_embed: torch.Tensor       # [B, D] normalised enhanced visual embedding
    p_sem: torch.Tensor         # [C, D] normalised semantic prototypes
    attr_pred: torch.Tensor     # [B, K] slot attention peak (L_attr input)
    cls_anchor: torch.Tensor    # [B, D] frozen CLIP CLS (zero-shot anchor)
    slot_readout: torch.Tensor  # [B, K, D] slot read-out R
