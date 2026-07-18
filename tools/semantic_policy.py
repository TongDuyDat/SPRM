#!/usr/bin/env python
"""Shared semantic-source policy for train/eval (Plan 3 unify_after_llm)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def unify_after_llm_enabled(cfg) -> bool:
    experiment = getattr(cfg, "experiment", None)
    if experiment is None:
        return False
    return bool(getattr(experiment, "unify_after_llm", False))


def resolve_semantic_source(cfg) -> str:
    """Resolve data.semantic_source; force class_desc when unify_after_llm."""
    data_cfg = cfg.data
    if unify_after_llm_enabled(cfg):
        requested = getattr(data_cfg, "semantic_source", "auto")
        if requested is not None:
            requested = str(requested).lower()
        if requested not in {None, "auto", "class_desc"}:
            raise ValueError(
                "experiment.unify_after_llm=true requires data.semantic_source "
                f"in {{auto, class_desc}}; got {requested!r}."
            )
        return "class_desc"

    source = getattr(data_cfg, "semantic_source", "auto")
    if source is None:
        source = "auto"
    source = str(source).lower()
    if source == "auto":
        if getattr(data_cfg, "semantic_embedding_path", None):
            return "embedding"
        if getattr(data_cfg, "semantic_metadata_path", None):
            return "class_desc"
        return "attribute"
    allowed = {"attribute", "embedding", "class_desc"}
    if source not in allowed:
        raise ValueError(
            f"Unknown data.semantic_source={source!r}. Expected one of {sorted(allowed)}."
        )
    return source


def configure_semantic_input(cfg, attribute, semantic_source: str) -> None:
    """Mutate cfg.model.semantic for the chosen semantic source."""
    semantic_cfg = cfg.model.semantic
    if semantic_source == "class_desc":
        backbone_type = getattr(semantic_cfg, "backbone_type", "embedding")
        if backbone_type in {"attribute", "identity", "embedding"}:
            raise ValueError(
                "semantic_source='class_desc' requires a text semantic backbone "
                "such as 'clip' or 'bert', not backbone_type='embedding'."
            )
        if getattr(semantic_cfg, "input_dim", None) is not None:
            raise ValueError(
                "semantic_source='class_desc' must keep model.semantic.input_dim null. "
                "Raw name_c/attr_c/hier_c/desc_c records are encoded by the text backbone."
            )
        shared_dim = getattr(semantic_cfg, "shared_dim", None)
        out_dim = getattr(cfg.model.aggregator, "out_dim", None)
        if shared_dim != out_dim:
            raise ValueError(
                "Full HSPRN requires model.semantic.shared_dim == "
                "model.aggregator.out_dim. Got "
                f"shared_dim={shared_dim}, out_dim={out_dim}."
            )
        if unify_after_llm_enabled(cfg):
            # Prefer itemwise pool so long multi-item lists stay within CLIP budget.
            current = getattr(semantic_cfg, "itemwise_pool", None)
            if current is None:
                semantic_cfg.itemwise_pool = True
            elif current is not False:
                semantic_cfg.itemwise_pool = True
        return

    cfg.model.semantic.input_dim = int(attribute.shape[1])
    if getattr(cfg.model.loss, "use_full", True):
        shared_dim = getattr(cfg.model.semantic, "shared_dim", None)
        if shared_dim != cfg.model.semantic.input_dim:
            print(
                "WARNING: Eq. (33) semantic consistency requires "
                f"shared_dim ({shared_dim}) == semantic input dim "
                f"({cfg.model.semantic.input_dim}). "
                "The loss will raise unless config is adjusted."
            )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return len(value) == 0
    except TypeError:
        return False


def assert_full_after_llm_semantics(
    class_semantics: Sequence[Mapping],
    *,
    num_total_classes: int,
    context: str = "class_semantics",
) -> None:
    """Raise if any class is missing name_c/attr_c/hier_c/desc_c content."""
    if class_semantics is None:
        raise ValueError(
            f"{context}: unify_after_llm / class_desc requires loaded class semantics."
        )
    if len(class_semantics) != num_total_classes:
        raise ValueError(
            f"{context}: class semantic record count mismatch: got "
            f"{len(class_semantics)}, expected {num_total_classes}."
        )
    required = ("name_c", "attr_c", "hier_c", "desc_c")
    for index, record in enumerate(class_semantics):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"{context}: record #{index} must be a mapping, got {type(record)}."
            )
        missing = []
        for key in required:
            value = record.get(key)
            if value is None:
                # also accept short aliases already canonicalized by loaders
                aliases = {
                    "name_c": ("name", "class_name", "label"),
                    "attr_c": ("attr", "attributes", "attrs", "attribute"),
                    "hier_c": ("hier", "hierarchy", "parents", "path"),
                    "desc_c": ("desc", "description", "text", "caption"),
                }
                for alt in aliases[key]:
                    if alt in record and not _is_missing(record[alt]):
                        value = record[alt]
                        break
            if _is_missing(value):
                missing.append(key)
        if missing:
            name = record.get("name_c") or record.get("name") or f"index={index}"
            raise ValueError(
                f"{context}: class #{index} ({name!r}) missing after-LLM fields: "
                f"{missing}. Generate data_semantics/<DS>/class_descriptions_"
                f"<ds>_after_llm.json and rebuild the prepared package."
            )


def after_llm_filename(dataset_name: str) -> str:
    ds = str(dataset_name).strip().lower()
    return f"class_descriptions_{ds}_after_llm.json"


def after_llm_path(semantic_root: str | Any, dataset_name: str) -> str:
    from pathlib import Path

    root = Path(semantic_root)
    ds_upper = str(dataset_name).strip().upper()
    return str(root / ds_upper / after_llm_filename(ds_upper))


def is_legacy_class_descriptions_path(path: str | Any) -> bool:
    """True if path is the legacy class_descriptions_<ds>.json (no after_llm)."""
    from pathlib import Path

    name = Path(path).name.lower()
    if not name.startswith("class_descriptions_") or not name.endswith(".json"):
        return False
    if name.endswith("_after_llm.json"):
        return False
    # class_descriptions_apy.json / class_descriptions_cub.json
    return True
