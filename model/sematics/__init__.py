from .extractor import SemanticEmbeddingExtractor
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
            self.embedder = BertEmbedder(**kwargs)
            self.out_features = cfg["hidden_size"]
            self.tokenizer = self.embedder.tokenizer
            self.bert = self.embedder.model

        def forward(self, x):
            if isinstance(x, str):
                x = [x]
            if isinstance(x, list):
                x = self.tokenize(x)
            input_ids, mask = x
            return self.embedder.get_bert_embeddings(input_ids, mask)

        def _apply(self, fn):
            super()._apply(fn)
            for param in self.bert.parameters():
                self.embedder.device = param.device
                break
            return self

        def tokenize(self, texts):
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

def _build_tinyclip(pretrained=True, **kwargs):
    if not pretrained:
        raise ValueError("TinyCLIP semantic backbone requires pretrained text weights.")
    semantic_cfg = kwargs.get("semantic_cfg")
    model_name = _cfg_get(
        semantic_cfg,
        "pretrained_weights",
        "./pre-trained/text/tinyclip-text-3m",
    )

    class TinyCLIPTextWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            from transformers import AutoTokenizer, CLIPTextModelWithProjection

            self.model = CLIPTextModelWithProjection.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.out_features = int(self.model.config.projection_dim)
            self.max_text_len = min(
                int(_cfg_get(semantic_cfg, "max_text_len", 77)),
                int(self.model.config.max_position_embeddings),
            )

        def forward(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            encoded = self.tokenizer(
                list(texts),
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            return self.model(**encoded).text_embeds.float()

        def _apply(self, fn):
            super()._apply(fn)
            for param in self.model.parameters():
                if param.device.type == "cpu":
                    self.model.float()
                elif param.device.type == "cuda":
                    self.model.half()
                break
            return self

    return TinyCLIPTextWrapper()

def _build_sbert(pretrained=True, **kwargs):
    if not pretrained:
        raise ValueError("Sentence-BERT semantic backbone requires pretrained weights.")
    semantic_cfg = kwargs.get("semantic_cfg")
    model_name = _cfg_get(
        semantic_cfg,
        "pretrained_weights",
        "./pre-trained/text/sbert-minilm-l3",
    )

    class SentenceBERTWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            from transformers import AutoModel, AutoTokenizer

            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.out_features = int(self.model.config.hidden_size)
            self.max_text_len = min(
                int(_cfg_get(semantic_cfg, "max_text_len", 128)),
                int(self.model.config.max_position_embeddings),
            )

        def forward(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            encoded = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            token_embeddings = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
            pooled = (token_embeddings * mask).sum(dim=1)
            pooled = pooled / mask.sum(dim=1).clamp_min(1)
            return pooled.float()

        def _apply(self, fn):
            super()._apply(fn)
            for param in self.model.parameters():
                if param.device.type == "cpu":
                    self.model.float()
                elif param.device.type == "cuda":
                    self.model.half()
                break
            return self

    return SentenceBERTWrapper()

_SEMANTIC_REGISTRY = {
    "attribute": _build_identity,
    "identity": _build_identity,
    "embedding": _build_identity,
    "bert": _build_bert,
    "clip": _build_clip,
    "tinyclip": _build_tinyclip,
    "sbert": _build_sbert,
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
