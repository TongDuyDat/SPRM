from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch


class SemanticEmbeddingExtractor:
    """Build offline text prompts from per-class semantic metadata.

    The expected raw input is a class-level semantic record matching Eq. 4:
    {"name": ..., "attributes": ..., "hierarchy": ..., "description": ...}.
    All four semantic sources are required for training. Training should
    consume the cached tensor saved by tools/extract_semantic_embeddings.py,
    not raw metadata.
    """

    def __init__(
        self,
        backend: str = "bert",
        model_name: Optional[str] = None,
        max_length: Optional[int] = None,
        tokenizer: Any = None,
    ):
        self.backend = backend.lower()
        self.model_name = model_name or self._default_model_name(self.backend)
        self.max_length = max_length or self._default_max_length(self.backend)
        self.tokenizer = tokenizer

    @staticmethod
    def _default_model_name(backend: str) -> str:
        if backend == "clip":
            return "ViT-B/32"
        return "bert-base-uncased"

    @staticmethod
    def _default_max_length(backend: str) -> int:
        if backend == "clip":
            return 77
        return 128

    def build_prompts(self, class_semantics: Any) -> list[str]:
        """Convert one or more t_c records into text prompts."""
        records = self._as_records(class_semantics)
        return [self.build_prompt(record) for record in records]

    def build_prompt(self, class_semantic: Mapping[str, Any]) -> str:
        """Convert one t_c record into a single text prompt."""
        name = self._first_present(class_semantic, "name_c", "name", "class_name", "label")
        attrs = self._first_present(
            class_semantic, "attr_c", "attributes", "attrs", "attr", "attribute"
        )
        hierarchy = self._first_present(
            class_semantic, "hier_c", "hierarchy", "hier", "parents", "path"
        )
        description = self._first_present(
            class_semantic, "desc_c", "description", "desc", "text", "caption"
        )
        missing = []
        for field_name, value in (
            ("name_c", name),
            ("attr_c", attrs),
            ("hier_c", hierarchy),
            ("desc_c", description),
        ):
            if self._is_missing(value):
                missing.append(field_name)
        if missing:
            raise ValueError(
                "Each class semantic record must contain name_c, attr_c, hier_c, "
                f"and desc_c. Missing: {missing}"
            )
        attr_names = self._first_present(class_semantic, "attribute_names", "attr_names")

        parts = []
        if name:
            parts.append(f"Class name: {self._clean_text(name)}.")

        attr_text = self._format_attributes(attrs, attr_names)
        if attr_text:
            parts.append(f"Attributes: {attr_text}.")

        hierarchy_text = self._format_sequence(hierarchy, separator=" > ")
        if hierarchy_text:
            parts.append(f"Hierarchy: {hierarchy_text}.")

        if description:
            parts.append(f"Description: {self._clean_text(description)}.")

        if not parts:
            raise ValueError("Semantic class record must contain at least one t_c field.")
        return " ".join(parts)

    def tokenize(self, class_semantics: Any):
        """Tokenize t_c prompts for the selected text encoder."""
        prompts = self.build_prompts(class_semantics)
        if self.backend == "clip":
            from .clip import tokenize

            return tokenize(prompts, context_length=self.max_length, truncate=True)

        if self.backend == "bert":
            tokenizer = self.tokenizer or self._load_bert_tokenizer()
            encoded = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return encoded["input_ids"], encoded["attention_mask"]

        raise ValueError(
            f"Raw t_c metadata can only be tokenized for BERT or CLIP, got '{self.backend}'."
        )

    @torch.no_grad()
    def extract(self, class_semantics: Any, encoder, device: Optional[torch.device] = None):
        """Extract phi(t_c) with an already constructed encoder."""
        encoder_input = self.tokenize(class_semantics)

        if device is None:
            try:
                device = next(encoder.parameters()).device
            except (AttributeError, StopIteration):
                device = torch.device("cpu")

        if isinstance(encoder_input, tuple):
            encoder_input = tuple(x.to(device) for x in encoder_input)
        else:
            encoder_input = encoder_input.to(device)

        return encoder(encoder_input)

    def _load_bert_tokenizer(self):
        from transformers import BertTokenizer

        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_name, do_lower_case="uncased" in self.model_name
        )
        return self.tokenizer

    @staticmethod
    def _as_records(class_semantics: Any) -> list[Mapping[str, Any]]:
        if isinstance(class_semantics, Mapping):
            return [class_semantics]
        if isinstance(class_semantics, Sequence) and not isinstance(class_semantics, str):
            if not class_semantics:
                return []
            if all(isinstance(item, Mapping) for item in class_semantics):
                return list(class_semantics)
        raise TypeError(
            "class_semantics must be a dict or a list of dicts containing t_c fields."
        )

    @staticmethod
    def _first_present(record: Mapping[str, Any], *keys: str):
        for key in keys:
            if key in record and record[key] is not None:
                return record[key]
        return None

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        try:
            return len(value) == 0
        except TypeError:
            return False

    def _format_attributes(self, attrs: Any, attr_names: Any = None) -> str:
        if attrs is None:
            return ""

        if isinstance(attrs, Mapping):
            selected = []
            for key, value in attrs.items():
                if self._is_positive_attribute(value):
                    selected.append(self._clean_text(key))
            return ", ".join(selected)

        if isinstance(attrs, str):
            return self._clean_text(attrs)

        if torch.is_tensor(attrs):
            attrs = attrs.detach().cpu().tolist()

        if isinstance(attrs, Sequence):
            if all(isinstance(item, str) for item in attrs):
                return ", ".join(self._clean_text(item) for item in attrs)

            if attr_names is not None:
                return self._format_numeric_attributes(attrs, attr_names)

        return ""

    def _format_numeric_attributes(self, attrs: Sequence[Any], attr_names: Sequence[str]) -> str:
        pairs = []
        for value, name in zip(attrs, attr_names):
            if self._is_positive_attribute(value):
                pairs.append((float(value), self._clean_text(name)))

        pairs.sort(reverse=True)
        return ", ".join(name for _, name in pairs)

    def _format_sequence(self, value: Any, separator: str = ", ") -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return self._clean_text(value)
        if isinstance(value, Mapping):
            return separator.join(self._clean_text(v) for v in value.values() if v)
        if isinstance(value, Sequence):
            return separator.join(self._clean_text(v) for v in value if v)
        return self._clean_text(value)

    @staticmethod
    def _clean_text(value: Any) -> str:
        return str(value).replace("_", " ").strip()

    @staticmethod
    def _is_positive_attribute(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return bool(value.strip())
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return value is not None
