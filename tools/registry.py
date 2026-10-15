"""Minimal registry for config-selected components (OpenMMLab-style dispatch).

The config names a class by string and the entry points look it up here:

    data:  {type: "MultilabelSplit", dataset: "coco", ...}
    model: {loss: {type: "AsymmetricLoss", ...}}
    eval:  {type: "MultilabelTopK"}

This follows the dict-registry idiom already used in the repo
(``datasets/zsl_dataset.py`` DATASET_REGISTRY, ``model/vision/__init__.py``
_BACKBONE_REGISTRY). It is deliberately not ``mmengine``: that package is not
in requirements.txt, and the whole point is to remove a source of setup
friction rather than add one.

A registry maps a short name to a builder callable. Registering the same name
twice is an error, so a typo fails loudly at import rather than silently
selecting the wrong component.
"""

from __future__ import annotations


class Registry:
    """Maps a config string to a builder callable.

    Usage:
        DATASETS = Registry("dataset builder")

        @DATASETS.register("MultilabelSplit")
        def build_multilabel_split(cfg, ...):
            ...
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._builders: dict[str, object] = {}

    def register(self, name: str):
        def _decorator(fn):
            if name in self._builders:
                raise KeyError(
                    f"{self.kind} {name!r} is already registered "
                    f"(by {self._builders[name]})."
                )
            self._builders[name] = fn
            return fn

        return _decorator

    def build(self, name: str, *args, **kwargs):
        if name not in self._builders:
            raise KeyError(
                f"Unknown {self.kind} {name!r}. "
                f"Available: {sorted(self._builders)}"
            )
        return self._builders[name](*args, **kwargs)

    def get(self, name: str):
        """Return the builder without calling it (for introspection)."""
        if name not in self._builders:
            raise KeyError(
                f"Unknown {self.kind} {name!r}. "
                f"Available: {sorted(self._builders)}"
            )
        return self._builders[name]

    def names(self) -> list[str]:
        return sorted(self._builders)

    def __contains__(self, name: str) -> bool:
        return name in self._builders

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.names()})"


#: Dataset builders keyed by ``data.type`` in the config.
DATASETS = Registry("dataset builder")

#: Metric protocols keyed by ``eval.type`` in the config.
METRICS = Registry("metric protocol")

#: Loss modules keyed by ``model.loss.type`` in the config.
LOSSES = Registry("loss")

#: Trainer classes keyed by ``trainer.type`` in the config (one per benchmark
#: family: classic / CZSL / multi-label).
TRAINERS = Registry("trainer")
