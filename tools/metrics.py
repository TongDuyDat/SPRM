"""Metric protocols, one per benchmark family (chosen by the trainer / tools/eval.py).

This module is a THIN ADAPTER. It does not implement any metric formula. Every
number comes from one of four existing modules, which are the single source of
truth for the reported results and must not be rewritten:

    tools/evaluator.py            ZSLEvaluator       -> classic ZSL/GZSL Top-1
    tools/ml_evaluator.py         top-k F1, mAP      -> multi-label (doc A.4)
    tools/czsl_exact_metrics.py   closed_world_metrics -> CZSL exact sweep (A.3)
    tools/czsl_evaluator.py       CZSLEvaluator      -> CZSL 20-bin bias sweep

Why the adapter exists: the multi-label in-training GZSL bug
(``train_nuswide_plan.py:72`` vs ``:74``, and four more copies) happened because
each call site built its own zsl/gzsl slices inline, and one of them sliced the
same tensors twice. Here each protocol declares its slices ONCE, explicitly, and
``zsl_labels``/``gzsl_labels`` are named so a reviewer can see they differ. The
protocol also asserts ``zsl != gzsl`` for multi-label, so the old bug cannot
return silently.

Doc reference: ``docs/BAO_CAO_DANH_GIA_va_SO_SANH_V14_BENCH6.md`` A.3 / A.4 / A.6.
"""

from __future__ import annotations

import numpy as np
import torch

from tools.registry import METRICS


class BaseMetric:
    """Common protocol interface.

    ``compute(scores, targets)`` returns a flat ``dict`` of python floats. The
    caller (``tools/eval.py``) adds provenance fields; nothing here reads files
    or does IO.
    """

    name = "base"

    #: Which doc metrics this protocol is responsible for producing. Checked by
    #: the coverage gate (Gate 5) -- an empty cell fails.
    provides: tuple = ()

    def compute(self, scores, targets, **kw):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Multi-label (doc A.4) -- COCO, NUS-WIDE
# ---------------------------------------------------------------------------

@METRICS.register("MultilabelTopK")
class MultilabelTopK(BaseMetric):
    """Doc A.4: per-column ranking mAP, plus top-k P/R/F1, plus F1 at tau=median.

    ``scores`` and ``targets`` are [N, C] with C = the full eval space
    (65 for COCO, 1006 for NUS). ``unseen_pos`` are the column indices of the
    unseen classes inside that space.

    Slices (declared once, used for every number below):
      ZSL  : columns = unseen_pos,            rows = images with >=1 unseen positive
      GZSL : columns = all C,                 rows = images with >=1 positive
    For NUS these genuinely differ (81 classes / a subset of rows vs 1006
    classes / a superset of rows). If they ever coincide the run is wrong, so
    the protocol raises rather than reporting ZSL twice under a GZSL label.
    """

    name = "MultilabelTopK"
    provides = (
        "mAP_unseen", "mAP_all",
        "zsl_top3", "zsl_top5", "gzsl_top3", "gzsl_top5",
        "f1_unseen_median",
    )

    def __init__(self, unseen_pos=None, k_values=(3, 5), check_distinct=True):
        self.unseen_pos = None if unseen_pos is None else list(map(int, unseen_pos))
        self.k_values = tuple(int(k) for k in k_values)
        self.check_distinct = bool(check_distinct)

    def compute(self, scores, targets, **kw):
        from tools.ml_evaluator import (compute_topk_f1, f1_at_threshold,
                                        mean_average_precision)

        scores = self._as_numpy(scores)
        targets = self._as_numpy(targets)
        if scores.shape != targets.shape:
            raise ValueError(
                f"MultilabelTopK needs scores and targets the same shape, got "
                f"{scores.shape} and {targets.shape}."
            )
        C = scores.shape[1]
        up = self.unseen_pos if self.unseen_pos is not None else list(range(C))
        if up and (min(up) < 0 or max(up) >= C):
            raise ValueError(
                f"unseen_pos out of range for {C} columns: min={min(up)} max={max(up)}."
            )

        out = {}

        # mAP per doc A.4: unseen over unseen columns, all over all columns.
        # Not row-masked, matching tools/eval_nuswide_gzsl_1006.py:167-168.
        out["mAP_unseen"] = float(mean_average_precision(scores, targets, up) * 100)
        out["mAP_all"] = float(mean_average_precision(scores, targets) * 100)

        # --- ZSL slice: unseen columns, images with >=1 unseen positive ---
        s_u = scores[:, up]
        t_u = targets[:, up]
        mask_zsl = t_u.sum(axis=1) > 0
        # --- GZSL slice: all columns, images with >=1 positive ---
        mask_gzsl = targets.sum(axis=1) > 0

        out["n_images_zsl"] = int(mask_zsl.sum())
        out["n_images_gzsl"] = int(mask_gzsl.sum())
        out["n_classes_zsl"] = int(len(up))
        out["n_classes_gzsl"] = int(C)

        for k in self.k_values:
            out[f"zsl_top{k}"] = compute_topk_f1(s_u[mask_zsl], t_u[mask_zsl], k_val=k)
            out[f"gzsl_top{k}"] = compute_topk_f1(scores[mask_gzsl],
                                                 targets[mask_gzsl], k_val=k)

        # F1 at the median score (doc A.4: cosine scores do not sit around 0.5).
        thr = float(np.median(scores))
        f1u = f1_at_threshold(scores, targets, up, thr=thr)
        f1u["threshold"] = thr
        out["f1_unseen_median"] = f1u

        # Degeneracy guard. GZSL is only defined when the scoring space contains
        # BOTH seen and unseen columns: over an all-unseen space "generalized"
        # means nothing and the GZSL number is ZSL wearing a different label.
        # That is exactly the NUS in-training bug (train_nuswide_plan.py:67-75,
        # eval_bench6.py:220-222, ...), where the evaluator only ever had 81
        # unseen columns. Raise rather than report a meaningless number.
        if self.check_distinct:
            if len(up) >= C:
                raise AssertionError(
                    f"MultilabelTopK: the scoring space has {C} columns and all "
                    f"{len(up)} of them are unseen, so there is no seen class to "
                    "generalize over. GZSL is undefined here -- returning this "
                    "as GZSL would be reporting ZSL twice (the NUS bug). Pass "
                    "the full seen+unseen space (65 for COCO, 1006 for NUS)."
                )
            z, g = out["zsl_top3"]["f1"], out["gzsl_top3"]["f1"]
            if z == g:
                raise AssertionError(
                    "MultilabelTopK: zsl_top3 == gzsl_top3 "
                    f"({z:.6f}) with {len(up)}/{C} unseen columns. ZSL and GZSL "
                    "are computed from different slices and must differ; "
                    "reporting ZSL under a GZSL label is the NUS bug."
                )
        return out

    @staticmethod
    def _as_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Classic ZSL / GZSL (doc A.6) -- CUB, AWA2, APY, SUN
# ---------------------------------------------------------------------------

@METRICS.register("ZSLPoint")
class ZSLPoint(BaseMetric):
    """Doc A.6: point-based Top-1 per-class, GZSL u/s/H, calibrated stacking.

    Expects the trainer's convention: ``scores_seen``/``scores_unseen`` are
    [N, C_total] with -inf outside the candidate set, and labels are class
    indices. Delegates entirely to ``ZSLEvaluator``.
    """

    name = "ZSLPoint"
    provides = ("zsl_T1", "zsl_T1_seen", "gzsl_u", "gzsl_s", "gzsl_H",
                "cal_u", "cal_s", "cal_H", "cal_gamma", "hub_seen", "hub_unseen")

    def __init__(self, seen_classes, unseen_classes, calibrate=True):
        from tools.evaluator import ZSLEvaluator

        self.seen_classes = np.asarray(seen_classes)
        self.unseen_classes = np.asarray(unseen_classes)
        self.evaluator = ZSLEvaluator(self.seen_classes, self.unseen_classes)
        self.calibrate = bool(calibrate)

    def compute(self, scores_seen, labels_seen, scores_unseen, labels_unseen, **kw):
        ev = self.evaluator
        out = {
            "zsl_T1": float(ev.eval_zsl(scores_unseen, labels_unseen) * 100),
            "zsl_T1_seen": float(ev.eval_zsl_seen(scores_seen, labels_seen) * 100),
        }
        u, s, H = ev.eval_gzsl(scores_seen, labels_seen, scores_unseen, labels_unseen)
        out.update({"gzsl_u": float(u * 100), "gzsl_s": float(s * 100),
                    "gzsl_H": float(H * 100)})
        if self.calibrate:
            cu, cs, cH, gamma = ev.eval_gzsl_calibrated(
                scores_seen, labels_seen, scores_unseen, labels_unseen)
            out.update({"cal_u": float(cu * 100), "cal_s": float(cs * 100),
                        "cal_H": float(cH * 100), "cal_gamma": float(gamma)})
        hub = ev.hubness_stats(scores_seen, scores_unseen)
        out.update({"hub_seen": float(hub["hub_seen_ratio"]),
                    "hub_unseen": float(hub["hub_unseen_ratio"])})
        return out


# ---------------------------------------------------------------------------
# CZSL (doc A.3) -- MIT-States, UT-Zappos, C-GQA
# ---------------------------------------------------------------------------

@METRICS.register("CZSLSweep")
class CZSLSweep(BaseMetric):
    """Doc A.3: full Top-1 bias curve -> Best Seen/Unseen/HM, AUC, and the
    diagnostic accuracies.

    Emits BOTH bias sweeps, from one scores array (pure post-processing, no
    re-inference):
      ``auc_exact``  -- tools/czsl_exact_metrics.py, every score-gap breakpoint
      ``auc_20bin``  -- tools/czsl_evaluator.py, published magic_binsize=20

    Doc A.3 names magic_bins=20; REPORT_PLAN_V14.md:27 describes the exact
    sweep; czsl_exact_metrics.py states outright that the two differ. Reporting
    both means neither claim has to be adjudicated and both figure sets stay
    reproducible.
    """

    name = "CZSLSweep"
    provides = ("best_seen", "best_unseen", "best_hm", "auc_exact", "auc_20bin",
                "hm_seen", "hm_unseen", "unbiased_seen_acc", "unbiased_unseen_acc",
                "unbiased_hm", "pair_acc", "attr_acc", "obj_acc",
                "obj_oracle_acc", "biasterm")

    def __init__(self, bench, universe="closed", want_20bin=True):
        self.bench = bench
        self.universe = universe
        self.want_20bin = bool(want_20bin)

    def compute(self, scores, pair_labels, **kw):
        from tools.czsl_exact_metrics import closed_world_metrics

        x = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
        y = (pair_labels.detach().cpu().numpy() if torch.is_tensor(pair_labels)
             else np.asarray(pair_labels))

        bench = self.bench
        if self.universe == "full":
            expected = {(a, o) for a in bench["attrs"] for o in bench["objs"]}
            if set(bench["pairs"]) != expected:
                raise ValueError(
                    "Open-world needs scores over the full attr x obj product."
                )
            candidates = expected
        else:
            candidates = set(bench["train_pairs"]) | set(bench["test_pairs"])

        exact = closed_world_metrics(x, y, bench["pairs"],
                                     set(bench["train_pairs"]), candidates)
        out = {
            "protocol_exact": exact["protocol"],
            "best_seen": exact["best_seen"] * 100,
            "best_unseen": exact["best_unseen"] * 100,
            "best_hm": exact["best_hm"] * 100,
            "hm_seen": exact["hm_seen"] * 100,
            "hm_unseen": exact["hm_unseen"] * 100,
            "unbiased_seen_acc": exact["unbiased_seen_acc"] * 100,
            "unbiased_unseen_acc": exact["unbiased_unseen_acc"] * 100,
            "unbiased_hm": exact["unbiased_hm"] * 100,
            "pair_acc": exact["pair_acc"] * 100,
            "attr_acc": exact["attr_acc"] * 100,
            "obj_acc": exact["obj_acc"] * 100,
            "obj_oracle_acc": exact["obj_oracle_acc"] * 100,
            "biasterm": exact["biasterm"],
            "auc_exact": exact["auc"] * 100,
            "auc_exact_raw": exact["auc"],
            "curve": exact["curve"],
        }

        if self.want_20bin:
            out["auc_20bin"] = self._sweep_20bin(x, y)
        return out

    def _sweep_20bin(self, scores, pair_labels):
        """Published 20-bin sweep via CZSLEvaluator (best-effort).

        Returns None when the legacy evaluator cannot accept this bench (e.g.
        the full-product open-world case); the exact sweep is authoritative and
        always present, so a missing 20-bin value is reported, not fabricated.
        """
        try:
            from tools.czsl_evaluator import CZSLEvaluator
            import torch as _t

            ev = CZSLEvaluator(self.bench)
            pairs_t = [(self.bench["pairs"][i][0], self.bench["pairs"][i][1])
                       for i in pair_labels.tolist()]
            attr_truth = _t.LongTensor([ev.attr2idx[a] for a, o in pairs_t])
            obj_truth = _t.LongTensor([ev.obj2idx[o] for a, o in pairs_t])
            pair_truth = _t.from_numpy(pair_labels).long()
            sc = _t.from_numpy(scores).clone()

            preds = ev.generate_predictions(sc, obj_truth)
            preds["scores"] = sc
            preds["pair_names"] = self.bench["pairs"]
            stats = ev.evaluate_predictions(preds, attr_truth, obj_truth,
                                            pair_truth, sc)
            return float(stats["AUC"]) * 100
        except Exception as exc:  # noqa: BLE001 - reported, never silently zero
            print(f"[CZSLSweep] auc_20bin unavailable: {type(exc).__name__}: {exc}")
            return None
