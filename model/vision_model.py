import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .vision import load_vision_backbone


class VisionModel(nn.Module):
    """
    Trích xuất đặc trưng thị giác với chọn mức (level) linh hoạt từ config.

    Cấu hình chọn mức:
      cfg.model.vision.levels: list[str]
        - ResNet: 'res1','res2','res3','res4' (alias cho 'layer1'..'layer4')
                   Có thể dùng trực tiếp 'layer1',...,'layer4'.
        - ViT: 'block_{k}' (k bắt đầu từ 1), 'tokens_final', 'cls'.
        - Nếu không chỉ định levels, tham chiếu cfg.feature_level: 'multilevel' sẽ
          tách 3 mức như trước; các mode 'patch'/'pool'/'both' cũng được hỗ trợ.

    Đầu ra:
      - Nếu có levels: trả list[Tensor] theo thứ tự yêu cầu (mỗi tensor có thể là
        [B,N,C] hoặc [B,C,H,W] hoặc [B,C]).
      - Nếu đầu vào x là [B,D] (feature đã trích sẵn): trả y nguyên dạng [B,D].
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        vision_cfg = cfg.model.vision
        backbone_kwargs = {}
        for key in ("clip_model", "download_root"):
            if hasattr(vision_cfg, key):
                backbone_kwargs[key] = getattr(vision_cfg, key)
        self.arch_type, self.backbone = load_vision_backbone(
            backbone_type=vision_cfg.backbone_type,
            pretrained=getattr(vision_cfg, 'pretrained', True),
            **backbone_kwargs,
        )
        self.feature_lv = getattr(cfg, 'feature_level', 'multilevel')
        self.levels = getattr(getattr(cfg.model, 'vision', object()), 'levels', None)
        self.freeze_backbone = bool(getattr(vision_cfg, 'freeze', False))
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    # ---------------------- ViT helpers ----------------------
    def _vit_tokens_after_blocks(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return tokens after all blocks + final LayerNorm
        tokens, x_mask, _ = self.backbone.forward_features(x)
        return tokens, x_mask

    def _vit_extract_levels(self, x, levels: List[str]) -> List[torch.Tensor]:
        x_tok, x_mask, _ = self.backbone.visual_embed(x)
        depth = len(self.backbone.blocks)
        out: List[torch.Tensor] = []
        wanted_blocks = {
            int(name.split('_')[1])
            for name in levels if name.startswith('block_') and name.split('_')[1].isdigit()
        }
        for i, blk in enumerate(self.backbone.blocks, start=1):
            x_tok, _ = blk(x_tok, mask=x_mask)
            if i in wanted_blocks:
                out.append(x_tok)
        # Post-norm tokens
        x_tok = self.backbone.norm(x_tok)
        if 'tokens_final' in levels:
            out.append(x_tok)
        if 'cls' in levels:
            out.append(x_tok[:, 0])  # [B, C]
        return out

    def _multilevel_vit(self, x):
        x_tok, x_mask, _ = self.backbone.visual_embed(x)
        depth = len(self.backbone.blocks)
        boundaries = {max(1, depth // 3), max(1, 2 * depth // 3), depth}
        multilevel = []
        for i, blk in enumerate(self.backbone.blocks, start=1):
            x_tok, _ = blk(x_tok, mask=x_mask)
            if i in boundaries:
                multilevel.append(x_tok)
        multilevel[-1] = self.backbone.norm(multilevel[-1])
        zl, zm, zh = multilevel
        return zl, zm, zh

    def _vit_patch_tokens(self, x):
        tokens, _, _ = self.backbone.forward_features(x)
        return tokens  # [B, N, C]

    def _vit_pool_token(self, x):
        tokens = self._vit_patch_tokens(x)
        return tokens[:, 0]  # [B, C]

    # -------------------- ResNet helpers ---------------------
    def _resnet_stage_outputs(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        res1 = self.backbone.layer1(x)
        res2 = self.backbone.layer2(res1)
        res3 = self.backbone.layer3(res2)
        res4 = getattr(self.backbone, 'layer4', None)
        if res4 is not None:
            res4 = self.backbone.layer4(res3)
        return res1, res2, res3, res4

    def _resnet_extract_levels(self, x, levels: List[str]) -> List[torch.Tensor]:
        name_map = {
            'res1': 'layer1', 'res2': 'layer2', 'res3': 'layer3', 'res4': 'layer4',
        }
        canonical = [name_map.get(lv, lv) for lv in levels]
        res1, res2, res3, res4 = self._resnet_stage_outputs(x)
        bank = {
            'layer1': res1, 'layer2': res2, 'layer3': res3,
        }
        if res4 is not None:
            bank['layer4'] = res4
        out = []
        for key in canonical:
            if key not in bank:
                raise ValueError(f"Requested ResNet level '{key}' is not available for this backbone.")
            out.append(bank[key])
        return out

    def _multilevel_resnet(self, x):
        res1, res2, res3, _ = self._resnet_stage_outputs(x)
        return res1, res2, res3

    # ------------------------- Forward -----------------------
    def forward(self, x: torch.Tensor):
        # Pre-extracted feature vectors pass-through
        if x.dim() == 2:
            return x

        if self.arch_type == 'clip':
            if isinstance(self.levels, (list, tuple)) and len(self.levels) > 0:
                wanted = []
                for name in self.levels:
                    if not (name.startswith("block_") and name.split("_")[1].isdigit()):
                        raise ValueError(
                            "CLIP visual backbone supports only block_<k> levels, "
                            f"got '{name}'."
                        )
                    wanted.append(int(name.split("_")[1]))
                return self.backbone.extract_levels(x, wanted)
            if self.feature_lv not in ('multilevel', 'pool'):
                raise ValueError(f"Feature level '{self.feature_lv}' is not supported for CLIP visual backbone.")
            return self.backbone(x)

        # Explicit levels from config
        if isinstance(self.levels, (list, tuple)) and len(self.levels) > 0:
            if self.arch_type == 'vit':
                return self._vit_extract_levels(x, list(self.levels))
            if self.arch_type == 'resnet':
                return self._resnet_extract_levels(x, list(self.levels))
            raise ValueError(f"Custom levels not supported for arch '{self.arch_type}'")

        # Backwards-compatible feature_level modes
        if self.feature_lv == 'multilevel':
            return self._multilevel_vit(x) if self.arch_type == 'vit' else self._multilevel_resnet(x)
        if self.feature_lv == 'patch':
            if self.arch_type == 'vit':
                return self._vit_patch_tokens(x)
            if self.arch_type == 'resnet':
                # tokens from last spatial map
                res1, res2, res3, _ = self._resnet_stage_outputs(x)
                B, C, H, W = res3.shape
                return res3.flatten(2).transpose(1, 2)
        if self.feature_lv == 'pool':
            if self.arch_type == 'vit':
                return self._vit_pool_token(x)
            if self.arch_type == 'resnet':
                _, _, res3, _ = self._resnet_stage_outputs(x)
                return F.adaptive_avg_pool2d(res3, 1).flatten(1)
        if self.feature_lv == 'both':
            if self.arch_type == 'vit':
                tokens = self._vit_patch_tokens(x)
                return tokens, tokens[:, 0]
            if self.arch_type == 'resnet':
                res1, res2, res3, _ = self._resnet_stage_outputs(x)
                tokens = res3.flatten(2).transpose(1, 2)
                return tokens, tokens.mean(dim=1)

        raise ValueError(f"Feature level '{self.feature_lv}' not supported")
