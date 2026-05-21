import torch
import torch.nn as nn
import torch.nn.functional as F
from . import load_semantic_backbone


class PriSematicConstructor(nn.Module):
    """
    4.2 Primitive Semantic Constructor

    Constructs semantic prototypes for all classes.

    Pipeline (per-class c):
        1. Gather semantic info  t_c = {name_c, attr_c, hier_c, desc_c}   (Eq. 4)
        2. Encode via text encoder  phi(t_c)                                (Eq. 5)
        3. Project into shared space  p_sem_c = W_s phi(t_c)               (Eq. 6)
        4. L2-normalize  ~p_sem_c = p_sem_c / ||p_sem_c||_2               (Eq. 7)

    Output:
        P_sem = {~p_1_sem, ..., ~p_C_sem}                                  (Eq. 8)
    """

    def __init__(self, cfg):
        super(PriSematicConstructor, self).__init__()
        self.cfg = cfg

        # --- Text encoder phi(.) (Eq. 5) ---
        backbone_type = getattr(cfg.model.semantic, 'backbone_type', 'bert')
        pretrained = getattr(cfg.model.semantic, 'pretrained', True)

        self.encoder = load_semantic_backbone(
            backbone_type=backbone_type,
            pretrained=pretrained
        )

        # Encoder output dimension
        self.encoder_dim = getattr(self.encoder, 'out_features', 768)

        # Target dimension in the shared semantic space
        self.shared_dim = getattr(cfg.model.semantic, 'shared_dim', 512)

        # --- Learnable projection W_s (Eq. 6) ---
        self.Ws = nn.Linear(self.encoder_dim, self.shared_dim, bias=False)

    def forward(self, text_feature):
        """
        Args:
            text_feature: Input to the text encoder.
                For BERT: tuple (input_ids, attention_mask).
                For attribute-based: tensor (C, attr_dim).
        Returns:
            p_sem_normalized: Normalized semantic prototype bank P_sem,
                              shape (C, shared_dim).  (Eq. 8)
        """
        # Eq. 5: phi(t_c) — encode semantic descriptions
        phi_t = self.encoder(text_feature)

        # Eq. 6: p_sem_c = W_s * phi(t_c)
        p_sem = self.Ws(phi_t)

        # Eq. 7: ~p_sem_c = p_sem_c / ||p_sem_c||_2
        p_sem_normalized = F.normalize(p_sem, p=2, dim=-1)

        return p_sem_normalized
