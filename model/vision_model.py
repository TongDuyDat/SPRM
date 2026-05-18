import torch
import torch.nn as nn
from .vision import load_vision_backbone


class VisionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.arch_type, self.backbone = load_vision_backbone(
            backbone_type=cfg.model.vision.backbone_type,
            pretrained=getattr(cfg.model.vision, 'pretrained', True),
        )
        self.feature_lv = cfg.feature_level

    # ------------------------------------------------------------------ #
    #  Multi-level feature extraction  Z = {zl, zm, zh}   (eq. 11)       #
    # ------------------------------------------------------------------ #

    def _multilevel_vit(self, x):
        """ViT: split transformer blocks into 3 equal groups."""
        x, x_mask, _, _ = self.backbone.visual_embed(x)

        depth = len(self.backbone.blocks)
        boundaries = {depth // 3, 2 * depth // 3, depth}

        multilevel = []
        for i, blk in enumerate(self.backbone.blocks):
            x, _ = blk(x, mask=x_mask)
            if (i + 1) in boundaries:
                multilevel.append(x)

        # LayerNorm only on final (high-level) features
        multilevel[-1] = self.backbone.norm(multilevel[-1])
        zl, zm, zh = multilevel
        return zl, zm, zh

    def _multilevel_resnet(self, x):
        """ResNet: use layer1/2/3 as low/mid/high-level features."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        zl = self.backbone.layer1(x)   # low-level   (e.g. 256ch)
        zm = self.backbone.layer2(zl)  # mid-level   (e.g. 512ch)
        zh = self.backbone.layer3(zm)  # high-level  (e.g. 1024ch)
        # layer4 omitted — too abstract for multi-level fusion;
        # include if needed: zh = self.backbone.layer4(zh)
        return zl, zm, zh

    def _extract_multilevel(self, x):
        if self.arch_type == 'vit':
            return self._multilevel_vit(x)
        elif self.arch_type == 'resnet':
            return self._multilevel_resnet(x)
        else:
            raise ValueError(f"Multilevel not supported for arch '{self.arch_type}'")

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor):
        if self.feature_lv == 'patch':
            x = self.backbone.forward_patch_tokens(x)
        elif self.feature_lv == 'pool':
            x = self.backbone.forward_pool_token(x)
        elif self.feature_lv == 'both':
            x = self.backbone.forward_both(x)
        elif self.feature_lv == 'multilevel':
            # Eq. (11): Z = {zl, zm, zh}
            return self._extract_multilevel(x)
        else:
            raise ValueError(f"Feature level '{self.feature_lv}' not supported")
        return x