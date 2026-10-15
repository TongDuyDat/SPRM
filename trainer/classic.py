"""Classic ZSL/GZSL trainer — CUB, AWA2, APY, SUN (doc A.6 family).

The v14 base already implements this family exactly: Tri-Loss, index labels,
point-based Top-1 via ``ZSLEvaluator``, ``final.pth`` at the end of the
schedule. This subclass exists so the benchmark families are symmetric and the
config names one explicitly:

    trainer: {type: "ClassicTrainer"}

Nothing is overridden — if that ever stops being true, this file is where the
classic-only behaviour belongs.
"""

from tools.registry import TRAINERS

from .base import TrainerBase


@TRAINERS.register("ClassicTrainer")
class ClassicTrainer(TrainerBase):
    """Inherits the whole v14 loop; see ``TrainerBase``."""
