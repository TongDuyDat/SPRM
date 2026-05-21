"""
ZSL / GZSL Evaluation following Xian et al. (TPAMI 2018).

Metrics:
    ZSL:  Top-1 per-class accuracy on unseen classes
    GZSL: u (unseen acc), s (seen acc), H = 2*u*s/(u+s) harmonic mean

Per-class accuracy = mean of per-class correct rates (handles class imbalance).
"""

import torch
import numpy as np


def compute_per_class_acc(y_true, y_pred, class_ids):
    """Compute mean per-class Top-1 accuracy.

    Args:
        y_true: (N,) ground-truth labels (original class indices).
        y_pred: (N,) predicted labels (original class indices).
        class_ids: array of class indices to evaluate over.

    Returns:
        float: mean per-class accuracy.
    """
    acc_per_class = []
    for c in class_ids:
        mask = (y_true == c)
        if mask.sum() == 0:
            continue
        correct = (y_pred[mask] == c).float().mean().item()
        acc_per_class.append(correct)

    if len(acc_per_class) == 0:
        return 0.0
    return float(np.mean(acc_per_class))


class ZSLEvaluator:
    """Evaluates ZSL and GZSL performance.

    Usage:
        evaluator = ZSLEvaluator(seen_classes, unseen_classes)
        # Predict scores for all test samples
        zsl_acc = evaluator.eval_zsl(scores_unseen, labels_unseen)
        u, s, H = evaluator.eval_gzsl(scores_seen, labels_seen,
                                        scores_unseen, labels_unseen)
    """

    def __init__(self, seen_classes, unseen_classes):
        """
        Args:
            seen_classes: np.array of seen class indices.
            unseen_classes: np.array of unseen class indices.
        """
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.all_classes = np.concatenate([seen_classes, unseen_classes])

    def eval_zsl(self, scores, labels):
        """Conventional ZSL: predict only among unseen classes.

        Args:
            scores: (N, C_total) compatibility scores for unseen test samples.
            labels: (N,) ground-truth labels for unseen test samples.

        Returns:
            float: Top-1 per-class accuracy on unseen classes.
        """
        # Mask seen classes to -inf so argmax only picks unseen
        masked_scores = scores.clone()
        for c in range(scores.shape[1]):
            if c not in self.unseen_classes:
                masked_scores[:, c] = -float("inf")

        preds = masked_scores.argmax(dim=1)
        return compute_per_class_acc(labels, preds, self.unseen_classes)

    def eval_gzsl(self, scores_seen, labels_seen, scores_unseen, labels_unseen):
        """Generalized ZSL: predict among all (seen + unseen) classes.

        Args:
            scores_seen: (N_s, C_total) scores for seen test samples.
            labels_seen: (N_s,) labels for seen test samples.
            scores_unseen: (N_u, C_total) scores for unseen test samples.
            labels_unseen: (N_u,) labels for unseen test samples.

        Returns:
            tuple: (u, s, H) — unseen acc, seen acc, harmonic mean.
        """
        # Predict over ALL classes (no masking)
        preds_seen = scores_seen.argmax(dim=1)
        preds_unseen = scores_unseen.argmax(dim=1)

        s = compute_per_class_acc(labels_seen, preds_seen, self.seen_classes)
        u = compute_per_class_acc(labels_unseen, preds_unseen, self.unseen_classes)

        if s + u == 0:
            H = 0.0
        else:
            H = 2 * u * s / (u + s)

        return u, s, H
