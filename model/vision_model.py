import torch.nn as nn

from .vision import CLIPVisualBackbone


class VisionModel(nn.Module):
    """Frozen CLIP ViT visual tower returning multi-level tokens and the CLS anchor.

    forward(x) -> {"tokens": [len(levels) x [B, N, 768]], "cls": [B, 512]}
    ``cfg.model.vision.levels`` must be ``block_<k>`` names.
    """

    def __init__(self, cfg):
        super().__init__()
        vc = cfg.model.vision
        self.backbone = CLIPVisualBackbone(vc.clip_model)
        levels = []
        for name in vc.levels:
            parts = str(name).split("_")
            if not (len(parts) == 2 and parts[0] == "block" and parts[1].isdigit()):
                raise ValueError(f"CLIP visual backbone supports only block_<k> levels, got '{name}'.")
            levels.append(int(parts[1]))
        self.levels = levels
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        return self.backbone.extract_levels_anchored(x, self.levels)
