import torch
import torch.nn as nn

from .vision_model import VisionModel
from .protypes import InstancePrototypeRefiner, RefinedPrototypeBank
from .sematics.constructor import PriSematicConstructor
from .aggregator import MultiFeatureAggregator
from .matching_head import SematicMatchingHead

class HSPRN(torch.nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(HSPRN, self).__init__(*args, **kwargs)
        self.cfg = cfg
        # Text Encoder
        self.psc = PriSematicConstructor(cfg)
        self.mfa = MultiFeatureAggregator(cfg)
        self.ipr = InstancePrototypeRefiner(cfg)
        
        # Vision Encoder
        self.vis_encoder = VisionModel(cfg)
        self.rp_bank = RefinedPrototypeBank(cfg)
        
        # Head
        self.msh = SematicMatchingHead(cfg)
        
    def forward(self, text_feature, vision_feature):
        # Text: PriSematicConstructor outputs P_sem (Eq. 8) directly
        p_sem = self.psc(text_feature)
        
        # Vision
        img_feat = self.vis_encoder(vision_feature)
        
        # Refinement and Aggregation
        ipr_out = self.ipr(p_sem, img_feat)
        rp_out = self.rp_bank(ipr_out)
        multi_vis_level = self.mfa(img_feat)
        
        # Matching (Final prediction)
        out = self.msh(multi_vis_level, rp_out)
        
        return out