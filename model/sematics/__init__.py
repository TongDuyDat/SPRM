from .bert_transformer import BertEmbedder
import torch.nn as nn

def _build_bert(pretrained=True, **kwargs):
    # Dummy default config matching original BertEmbedder requirements if not provided
    cfg = kwargs.get('config', {
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "max_text_len": 512,
        "drop_rate": 0.1
    })
    kwargs['config'] = cfg
    
    # We create a wrapper since BertEmbedder is not an nn.Module
    class BertWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            # BertEmbedder doesn't inherit from nn.Module in the original code,
            # but we need it to be part of the module tree if we want its weights saved
            self.embedder = BertEmbedder(**kwargs)
            # Expose out_features so projector knows the output dimension
            self.out_features = cfg["hidden_size"]

        def forward(self, x):
            """
            Expects x to be a tuple (input_ids, attention_mask)
            or handles raw text if we modify the constructor pipeline.
            """
            input_ids, mask = x
            return self.embedder.get_bert_embeddings(input_ids, mask)

    return BertWrapper()

_SEMANTIC_REGISTRY = {
    "bert": _build_bert,
    # "clip": _build_clip  # Can be added later
}

def load_semantic_backbone(backbone_type: str = 'bert', pretrained: bool = True, **kwargs):
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

__all__ = [
    "load_semantic_backbone",
    "BertEmbedder",
]
