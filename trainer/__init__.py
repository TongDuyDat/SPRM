"""Trainers: one per benchmark family, chosen by ``trainer.type`` in the config."""

from tools.registry import TRAINERS
from .base import TrainerBase
from .classic import ClassicTrainer
from .czsl import CZSLTrainer
from .multilabel import MultiLabelTrainer

__all__ = ["TrainerBase", "ClassicTrainer", "CZSLTrainer", "MultiLabelTrainer", "TRAINERS"]
