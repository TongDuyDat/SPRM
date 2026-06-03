from .extractor import SemanticEmbeddingExtractor
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

def _build_bert(pretrained=True, **kwargs):
    from .bert_transformer import BertEmbedder

    # Dummy default config matching original BertEmbedder requirements if not provided
    semantic_cfg = kwargs.pop("semantic_cfg", None)
    cfg = kwargs.get('config', {
        "vocab_size": 30522,
        "hidden_size": _cfg_get(semantic_cfg, "text_dim", 768),
        "num_layers": _cfg_get(semantic_cfg, "num_layers", 12),
        "num_heads": _cfg_get(semantic_cfg, "num_heads", 12),
        "mlp_ratio": 4,
        "max_text_len": _cfg_get(semantic_cfg, "max_text_len", 128),
        "drop_rate": 0.1
    })
    kwargs['config'] = cfg
    kwargs.setdefault("pretrained", pretrained)
    kwargs.setdefault(
        "pretrained_weights",
        _cfg_get(semantic_cfg, "pretrained_weights", "bert-base-uncased")
    )
    kwargs.setdefault("max_seq_len", _cfg_get(semantic_cfg, "max_text_len", 128))
    
    # We create a wrapper since BertEmbedder is not an nn.Module
    class BertWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            # BertEmbedder doesn't inherit from nn.Module in the original code,
            # but we need it to be part of the module tree if we want its weights saved
            self.embedder = BertEmbedder(**kwargs)
            # Expose out_features so projector knows the output dimension
            self.out_features = cfg["hidden_size"]
            self.tokenizer = self.embedder.tokenizer

        def forward(self, x):
            """
            Expects x to be a tuple (input_ids, attention_mask)
            or handles raw text if we modify the constructor pipeline.
            """
            input_ids, mask = x
            return self.embedder.get_bert_embeddings(input_ids, mask)

        def tokenize_texts(self, texts):
            return self.embedder.tokenize_texts(texts)

    return BertWrapper()

def _build_clip(pretrained=True, **kwargs):
    semantic_cfg = kwargs.get("semantic_cfg")
    model_name = _cfg_get(semantic_cfg, "pretrained_weights", None)
    if model_name is None:
        model_name = _cfg_get(semantic_cfg, "clip_model", "ViT-B/32")

    class CLIPTextWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            from .clip import load

            self.model, _ = load(model_name)
            self.out_features = int(self.model.text_projection.shape[1])

        def forward(self, x):
            device = next(self.model.parameters()).device
            return self.model.encode_text(x.to(device)).float()

    return CLIPTextWrapper()

_SEMANTIC_REGISTRY = {
    "attribute": _build_identity,
    "identity": _build_identity,
    "embedding": _build_identity,
    "bert": _build_bert,
    "clip": _build_clip,
}

def load_semantic_backbone(backbone_type: str = 'embedding', pretrained: bool = True, **kwargs):
    """Load a semantic backbone by name from config.

    Args:
        backbone_type: Key matching a registered semantic model, e.g. 'bert'.
        pretrained: Whether to load pretrained weights.
        **kwargs: Extra keyword arguments forwarded to the factory.

    Returns:
        nn.Module: The instantiated semantic backbone.
    """
    if backbone_type not in _SEMANTIC_REGISTRY:
        raise ValueError(
            f"Unknown semantic backbone_type '{backbone_type}'. "
            f"Available: {sorted(_SEMANTIC_REGISTRY.keys())}"
        )
    
    factory_fn = _SEMANTIC_REGISTRY[backbone_type]
    model = factory_fn(pretrained=pretrained, **kwargs)
    return model

def __getattr__(name):
    if name == "BertEmbedder":
        from .bert_transformer import BertEmbedder

        return BertEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_semantic_backbone",
    "BertEmbedder",
    "SemanticEmbeddingExtractor",
]
