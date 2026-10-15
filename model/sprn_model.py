import torch.nn as nn
import torch.nn.functional as F

from .aggregator import MultiFeatureAggregator
from .sprn_outputs import SPRNOutput
from .protypes import LatentPrototypeRefiner
from .sematics.constructor import PriSematicConstructor
from .vision_model import VisionModel


class SPRN(nn.Module):
    """SPRN with Latent Prototype Refinement (the only variant).

    Submodule names (psc, vis_encoder, mfa, W_local, ipr) are checkpoint keys.
    """

    def __init__(self, cfg, vis_encoder=None):
        super().__init__()
        self.cfg = cfg
        self.psc = PriSematicConstructor(cfg)
        # vis_encoder injection lets CPU tests bypass loading the CLIP tower.
        self.vis_encoder = vis_encoder if vis_encoder is not None else VisionModel(cfg)
        self.mfa = MultiFeatureAggregator(cfg)
        self.ipr = LatentPrototypeRefiner(cfg)
        d = cfg.model.aggregator.out_dim
        self.W_local = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.W_local.weight)

    def forward(self, text_feature, vision_feature):
        p_sem = self.psc(text_feature)["p_sem_norm"]
        visual = self.vis_encoder(vision_feature)
        cls_anchor = F.normalize(visual["cls"], p=2, dim=-1)
        v_global = self.mfa(visual)
        R, attr_pred, v_local = self.ipr(visual["tokens"][-1])
        v_tilde = F.normalize(v_global + self.W_local(v_local), p=2, dim=-1)
        scores = self.ipr.score(v_tilde, p_sem, R)
        return SPRNOutput(scores=scores, v_embed=v_tilde, p_sem=p_sem, attr_pred=attr_pred,
                           cls_anchor=cls_anchor, slot_readout=R)
