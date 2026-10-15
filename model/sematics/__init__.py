"""Semantic backbones: parameter-free embedding/identity (precomputed φ) or frozen CLIP text tower."""
import torch
import torch.nn as nn

def _cfg_get(cfg, name, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)

def _build_identity(pretrained=True, **kwargs):
    input_dim = kwargs.get("input_dim")
    semantic_cfg = kwargs.get("semantic_cfg")
    if input_dim is None:
        input_dim = _cfg_get(semantic_cfg, "text_dim", 768)

    class IdentitySemanticBackbone(nn.Module):
        def __init__(self, out_features):
            super().__init__()
            self.out_features = out_features

        def forward(self, x):
            return x.float()

    return IdentitySemanticBackbone(input_dim)

def _build_clip(pretrained=True, **kwargs):
    semantic_cfg = kwargs.get("semantic_cfg")
    model_name = _cfg_get(semantic_cfg, "pretrained_weights", None)
    if model_name is None:
        model_name = _cfg_get(semantic_cfg, "clip_model", "ViT-B/32")

    class CLIPTextWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            from .clip import load_text, tokenize

            self.model = load_text(model_name, device="cpu")
            self.out_features = int(self.model.text_projection.shape[1])
            self.context_length = _cfg_get(semantic_cfg, "max_text_len", 77)
            self._tokenize_fn = tokenize

        def forward(self, x):
            if isinstance(x, str):
                x = [x]
            if isinstance(x, (list, tuple)):
                x = self.tokenize(x)
            device = next(self.model.parameters()).device
            return self.model.encode_text(x.to(device)).float()

        def _apply(self, fn):
            super()._apply(fn)
            for param in self.model.parameters():
                if param.device.type == "cpu":
                    self.model = self.model.float()
                elif param.device.type == "cuda":
                    from .clip.model import convert_weights
                    convert_weights(self.model)
                break
            return self

        def tokenize(self, texts):
            return self._tokenize_fn(
                texts, context_length=self.context_length, truncate=True,
            )

    return CLIPTextWrapper()

_SEMANTIC_REGISTRY = {
    "identity": _build_identity,
    "embedding": _build_identity,
    "clip": _build_clip,
}


def load_semantic_backbone(backbone_type: str = 'embedding', pretrained: bool = True, **kwargs):
    """Load a semantic backbone by name from config."""
    if backbone_type not in _SEMANTIC_REGISTRY:
        raise ValueError(
            f"Unknown semantic backbone_type '{backbone_type}'. "
            f"Available: {sorted(_SEMANTIC_REGISTRY.keys())}"
        )

    factory_fn = _SEMANTIC_REGISTRY[backbone_type]
    return factory_fn(pretrained=pretrained, **kwargs)


__all__ = ["load_semantic_backbone"]
