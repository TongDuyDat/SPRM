from .vision_transformer_multitoken import *
from .vision_transformer_vit import *

import torch
import torch.nn as nn
import torchvision.models as tv_models


class CLIPVisualBackbone(nn.Module):
    """CLIP image encoder wrapper for HSPRN visual backbone registry.

    Only the visual tower is retained (the CLIP text branch is discarded), so
    the backbone's parameter count excludes the text encoder. Image features are
    computed exactly as CLIP's encode_image: ``visual(image.type(dtype))``.
    """

    def __init__(self, model_name="ViT-B/32", pretrained=True, **kwargs):
        super().__init__()
        if not pretrained:
            raise ValueError("CLIP visual backbone requires pretrained=True and a CLIP checkpoint/name.")

        download_root = kwargs.get("download_root", None)
        from ..sematics.clip import load

        clip_model, _ = load(model_name, device="cpu", jit=False, download_root=download_root)
        self.model_name = model_name
        # Keep only the visual tower; drop the CLIP text branch to save parameters.
        self.visual = clip_model.visual
        self.out_features = int(getattr(self.visual, "output_dim", 0))

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, x):
        out = self.visual(x.type(self.dtype))
        if isinstance(out, tuple):
            if len(out) >= 2 and out[1].dim() == 2:
                out = out[1]
            else:
                out = out[0]
                if out.dim() == 3:
                    out = out[:, 0]
        return out.float()

    def extract_levels(self, x, levels):
        """Return CLIP ViT block outputs as token maps [B, N, C].

        levels: iterable of 1-based block indices (e.g. [4, 8, 12] for the
        ViT-B/32 tower). Mirrors the multilevel mechanism used for ResNet
        (layer2/layer3/layer4) so the aggregator can combine three levels.
        """
        visual = self.visual
        dtype = self.dtype
        wanted = {int(level) for level in levels}
        x = x.type(dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_embedding = visual.class_embedding.to(dtype)
        x = torch.cat(
            [
                class_embedding
                + torch.zeros(
                    x.shape[0],
                    1,
                    class_embedding.shape[-1],
                    dtype=dtype,
                    device=x.device,
                ),
                x,
            ],
            dim=1,
        )
        x = x + visual.positional_embedding.to(dtype)
        x = visual.ln_pre(x)
        outs = []
        for index, block in enumerate(visual.transformer.resblocks, start=1):
            x = block(x)
            if index in wanted:
                outs.append(x.float())
        if len(outs) != len(wanted):
            depth = len(visual.transformer.resblocks)
            raise ValueError(
                f"Requested CLIP block levels {sorted(wanted)} but tower has "
                f"only {depth} blocks."
            )
        return outs

    def extract_levels_anchored(self, x, levels):
        """v10: like extract_levels, but also returns the bit-exact CLIP CLS
        embedding computed from the SAME forward pass.

        Standard CLIP encode_image is ``proj @ ln_post(final[:, 0])`` (see
        VisionTransformer.forward in model/sematics/clip/model.py). The
        extracted token maps above are pre-ln_post residual streams (768-d),
        so a CLS derived by pooling them would NOT match the embedding the
        R6b baseline scores with. Anchoring v10's epoch-0 output to that
        baseline requires both from one forward:

            cls = ln_post(x[:, 0, :]) @ proj        # [B, 512], bit-exact CLIP

        Returns {"tokens": [...(B, N, 768) maps...], "cls": (B, 512)}.
        """
        visual = self.visual
        dtype = self.dtype
        wanted = {int(level) for level in levels}
        x = x.type(dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_embedding = visual.class_embedding.to(dtype)
        x = torch.cat(
            [
                class_embedding
                + torch.zeros(
                    x.shape[0],
                    1,
                    class_embedding.shape[-1],
                    dtype=dtype,
                    device=x.device,
                ),
                x,
            ],
            dim=1,
        )
        x = x + visual.positional_embedding.to(dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND, mirroring VisionTransformer.forward
        outs = []
        for index, block in enumerate(visual.transformer.resblocks, start=1):
            x = block(x)
            if index in wanted:
                outs.append(x.permute(1, 0, 2).float())  # back to NLD [B, N, C]
        if len(outs) != len(wanted):
            depth = len(visual.transformer.resblocks)
            raise ValueError(
                f"Requested CLIP block levels {sorted(wanted)} but tower has "
                f"only {depth} blocks."
            )
        x = x.permute(1, 0, 2)  # LND -> NLD before taking the CLS token
        cls = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            cls = cls @ visual.proj
        return {"tokens": outs, "cls": cls.float()}


def _clip_visual(model_name):
    return lambda pretrained, **kw: CLIPVisualBackbone(
        model_name=kw.pop("clip_model", model_name),
        pretrained=pretrained,
        **kw,
    )


# Registry: backbone_type name -> (factory_fn, arch_type)
# arch_type: 'vit', 'resnet', or 'clip' — used to dispatch extraction
_BACKBONE_REGISTRY = {
    # --- ResNet ---
    "resnet50": ("resnet", lambda pretrained, **kw: tv_models.resnet50(pretrained=pretrained, **kw)),
    "resnet101": ("resnet", lambda pretrained, **kw: tv_models.resnet101(pretrained=pretrained, **kw)),
    # --- ViT ---
    "vit_small_patch16_224": ("vit", vit_small_patch16_224),
    "vit_base_patch16_224": ("vit", vit_base_patch16_224),
    "vit_base_patch32_224": ("vit", vit_base_patch32_224),
    "vit_base_patch16_384": ("vit", vit_base_patch16_384),
    "vit_base_patch32_384": ("vit", vit_base_patch32_384),
    "vit_large_patch16_224": ("vit", vit_large_patch16_224),
    "vit_large_patch32_224": ("vit", vit_large_patch32_224),
    "vit_large_patch16_384": ("vit", vit_large_patch16_384),
    "vit_large_patch32_384": ("vit", vit_large_patch32_384),
    # --- CLIP visual ViT ---
    "clip_vit_b32": ("clip", _clip_visual("ViT-B/32")),
    "clip_vit_b16": ("clip", _clip_visual("ViT-B/16")),
    "clip_vit_l14": ("clip", _clip_visual("ViT-L/14")),
}


def load_vision_backbone(backbone_type: str = 'vit_base_patch16_224', pretrained: bool = True, **kwargs):
    """Load a vision backbone by name from config.

    Args:
        backbone_type: Key matching a registered backbone,
                       e.g. 'vit_base_patch16_224' or 'resnet50'.
        pretrained: Whether to load pretrained weights (passed to factory).
        **kwargs: Extra keyword arguments forwarded to the factory function.

    Returns:
        tuple: (arch_type, model) where arch_type is 'vit' or 'resnet'.
    """
    if backbone_type not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone_type '{backbone_type}'. "
            f"Available: {sorted(_BACKBONE_REGISTRY.keys())}"
        )
    arch_type, factory_fn = _BACKBONE_REGISTRY[backbone_type]
    model = factory_fn(pretrained=pretrained, **kwargs)
    return arch_type, model


__all__ = [
    "load_vision_backbone",
    "CLIPVisualBackbone",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_base_patch32_224",
    "vit_base_patch16_384",
    "vit_base_patch32_384",
    "vit_large_patch16_224",
    "vit_large_patch32_224",
    "vit_large_patch16_384",
    "vit_large_patch32_384",
]
