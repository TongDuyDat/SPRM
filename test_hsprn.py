import torch

class MockConfig:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, MockConfig(v))
            else:
                setattr(self, k, v)

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.hsprn_model import HSPRN
from losses import HSPRNLoss

def test():
    cfg_dict = {
        'model': {
            'vision': {
                'backbone_type': 'vit_small_patch16_224',
                'pretrained': False
            },
            'semantic': {
                'backbone_type': 'embedding',
                'pretrained': False,
                'input_dim': 85,
                'text_dim': 85,
                'shared_dim': 512
            },
            'aggregator': {
                'dim_l': 768,
                'dim_m': 768,
                'dim_h': 768,
                'out_dim': 512
            },
            'loss': {
                'lambda1': 1.0,
                'lambda2': 0.1,
                'lambda3': 0.1,
                'lambda4': 0.1,
                'temperature': 0.1
            }
        },
        'feature_level': 'multilevel'
    }
    cfg = MockConfig(cfg_dict)

    print("Initializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HSPRN(cfg).to(device)
    
    B = 2
    C = 5
    
    vision_feature = torch.randn(B, 3, 224, 224).to(device)
    
    text_feature = torch.randn(C, 85).to(device)

    print("Running forward pass...")
    out = model(text_feature, vision_feature)
    
    print("Output shape:", out.shape)
    assert out.shape == (B, C), f"Output shape should be ({B}, {C}), got {out.shape}"
    
    print("All tests passed!")

if __name__ == "__main__":
    test()
