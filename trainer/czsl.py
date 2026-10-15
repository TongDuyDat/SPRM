"""Compositional ZSL trainer — MIT-States, UT-Zappos, C-QGA (doc A.3 family).

Training is identical to classic (``train_czsl.py`` states outright that it
keeps ``SPRN``, ``SPRNLoss`` and ``Trainer`` 100% unchanged and only feeds a
different dataset). What differs is the *evaluation protocol*: CZSL is scored
by a bias sweep over the (state, object) pair space, which chooses its own
operating point, so there is no calibrated-H / ZSL-T1 number to select a
checkpoint by — the published protocol is Best Seen/Unseen/HM from the curve.

Consequently this class overrides only the three evaluation hooks:

    trainer: {type: "CZSLTrainer"}

``evaluate()`` keeps the attribute-level diagnostics (pair/attr/obj accuracy)
that ``tools/czsl_exact_metrics.py`` also computes.
"""

import numpy as np

from tools.registry import TRAINERS

from .base import TrainerBase


@TRAINERS.register("CZSLTrainer")
class CZSLTrainer(TrainerBase):
    """v14 loop with the CZSL bias-sweep protocol for evaluation."""

    #: Which numbers the epoch line shows (see TrainerBase._log_eval).
    #: Evaluation protocol of this family. Fixed by the trainer, not by config.
    METRIC_TYPE = "CZSLSweep"

    _CZSL_LOG_KEYS = (
        ("best_hm", "BestHM"), ("best_seen", "BestSeen"),
        ("best_unseen", "BestUnseen"), ("pair_acc", "PairAcc"),
        ("attr_acc", "AttrAcc"), ("obj_acc", "ObjAcc"),
    )

    def _czsl_metric(self):
        """Lazily build the CZSL metric protocol from the config."""
        from tools.metrics import METRICS

        if not hasattr(self, "_czsl_metric_cache"):
            bench = self._czsl_bench()
            self._czsl_metric_cache = METRICS.build(
                self.METRIC_TYPE, bench=bench,
                universe="closed",
            )
        return self._czsl_metric_cache

    def _czsl_bench(self):
        """Pair-space bench for the metric (candidate set, pair names)."""
        from datasets.czsl_dataset import build_czsl_bench

        return build_czsl_bench(
            str(self.cfg.data.dataset),
            str(self.cfg.data.data_root),
            universe="closed",
        )

    def evaluate(self):
        """Score every test image against all pairs, then sweep the bias.

        Both test splits are scored in the full pair space and evaluated
        together, because a compositional classifier is scored over SEEN and
        UNSEEN pairs jointly (that is what the bias sweep trades between).
        """
        all_classes = list(range(self.num_total_classes))
        scores_seen, labels_seen = self._compute_scores(
            self.test_seen_loader, all_classes, "Eval seen")
        scores_unseen, labels_unseen = self._compute_scores(
            self.test_unseen_loader, all_classes, "Eval unseen")

        import torch

        scores = torch.cat([scores_seen, scores_unseen], dim=0).cpu()
        labels = torch.cat([labels_seen, labels_unseen], dim=0).cpu()
        return self._czsl_metric().compute(scores, labels)

    def _eval_stats(self):
        return self.evaluate()

    def _log_eval(self, stats):
        parts = []
        for key, label in self._CZSL_LOG_KEYS:
            if stats.get(key) is not None:
                parts.append(f"{label}: {stats[key]:.1f}%")
        auc = stats.get("auc_exact")
        if auc is not None:
            parts.append(f"AUC: {auc:.2f}")
        return " | " + " ".join(parts) if parts else ""

    def _selection(self, stats):
        """Select by Best HM, the headline CZSL number.

        Unlike classic there is no calibrated-H/ZSL-T1 pair, so this keeps
        ``best.pth`` only and drops the ``best_t1.pth`` slot.
        """
        hm = stats.get("best_hm")
        if hm is None:
            return []
        return [(hm, "best_H", "best.pth")]
