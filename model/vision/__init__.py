from .vision_transformer_multitoken import *
from .vision_transformer_vit import *

import torchvision.models as tv_models

# Registry: backbone_type name -> (factory_fn, arch_type)
# arch_type: 'vit' or 'resnet' — used to dispatch multilevel extraction
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