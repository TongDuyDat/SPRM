import torch
import torch.nn as nn
from . import load_semantic_backbone

class PriSematicConstructor(nn.Module):
    """
    4.2 Primitive Semantic Constructor
    Constructs semantic prototypes for all classes using a text encoder.
    """
    def __init__(self, cfg):
        super(PriSematicConstructor, self).__init__()
        self.cfg = cfg
        
        # Load semantic backbone from registry (e.g., 'bert', 'clip')
        backbone_type = getattr(cfg.model.semantic, 'backbone_type', 'bert')
        pretrained = getattr(cfg.model.semantic, 'pretrained', True)
        
        self.encoder = load_semantic_backbone(
            backbone_type=backbone_type,
            pretrained=pretrained
        )
        
        # We assume self.encoder provides `.out_features`.
        # If the output dimension of the text encoder doesn't match the target text_dim
        # expected by SematicBank (or if we want to ensure it's mapped correctly),
        # we can add an optional projection layer here.
        self.in_dim = getattr(self.encoder, 'out_features', 768)
        self.out_dim = getattr(cfg.model.semantic, 'text_dim', 768)
        
        if self.in_dim != self.out_dim:
            self.proj = nn.Linear(self.in_dim, self.out_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, text_feature):
        """
        Args:
            text_feature: Inputs to the text encoder.
                          For BERT, this might be a tuple (input_ids, attention_mask).
        Returns:
            phi_t: Encoded semantic features, shape (..., out_dim)
        """
        # Encode text \phi(t_c)
        encoded = self.encoder(text_feature)
        
        # Project if dimension mismatch
        phi_t = self.proj(encoded)
        
        return phi_t
