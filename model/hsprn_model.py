import torch
import torch.nn as nn
from typing import Any

from .vision_model import VisionModel
from .protypes import InstancePrototypeRefiner, RefinedPrototypeBank
from .sematics.constructor import PriSematicConstructor
from .aggregator import MultiFeatureAggregator
from .matching_head import SematicMatchingHead
from .hsprn_outputs import HSPRNOutput

class HSPRN(torch.nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(HSPRN, self).__init__(*args, **kwargs)
        self.cfg = cfg
        # Text / semantics
        self.psc = PriSematicConstructor(cfg)
        # Vision aggregator + refiner
        self.mfa = MultiFeatureAggregator(cfg)
        self.ipr = InstancePrototypeRefiner(cfg)
        self.vis_encoder = VisionModel(cfg)
        self.rp_bank = RefinedPrototypeBank(cfg)
        # Matching head
        self.msh = SematicMatchingHead(cfg)

        # Unified outputs for research-style losses
        self.return_full = getattr(getattr(cfg.model, 'loss', object()), 'return_full_output', True)

    def forward(self, text_feature, vision_feature):
        # Semantic branch
        # PriSematicConstructor returns (p_sem_norm, phi_t, p_sem_unnorm)
        p_sem_norm, phi_t_raw, p_sem_unnorm = self.psc(text_feature)

        # Vision branch (list/tuple of tensors or [B, D])
        vis_levels = self.vis_encoder(vision_feature)
        v_tilde = self.mfa(vis_levels)  # [B, d]

        # Refinement bank
        ipr_out = self.ipr(p_sem_norm, vis_levels)
        p_ref = self.rp_bank(ipr_out)  # [B, C, d]

        # Matching
        scores = self.msh(v_tilde, p_ref)  # [B, C]

        if self.return_full:
            return HSPRNOutput(
                scores=scores,
                v_embed=v_tilde,
                p_sem=p_sem_norm,
                p_ref=p_ref,
                phi_t=phi_t_raw,
                p_sem_unnorm=p_sem_unnorm,
            )
        return scores
