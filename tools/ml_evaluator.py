"""Multi-label metrics (mAP, F1) cho bench6 — protocol RCNn.

mAP: mean Average Precision trên unseen classes (17 COCO / 81 NUS),
tính theo cách chuẩn multi-label retrieval: mỗi lớp coi như 1 truy vấn,
rank ảnh theo score lớp đó.
"""

import numpy as np
import torch


def _ap(scores_cls, targets_cls):
    """Average Precision cho 1 lớp (scores_cls (N,), targets_cls (N,) 0/1)."""
    order = np.argsort(-scores_cls)
    y = targets_cls[order]
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    cumsum = np.cumsum(y)
    precision = cumsum / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / n_pos)


def mean_average_precision(scores, targets, cls_idx=None):
    """mAP trên các lớp cls_idx.

    Args:
        scores: (N, C) numpy/torch — score ảnh x lớp
        targets: (N, C) 0/1
        cls_idx: list int — chỉ tính mAP trên các lớp này (unseen protocol);
            None = tất cả.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    n_cls = scores.shape[1]
    if cls_idx is None:
        cls_idx = range(n_cls)
    aps = []
    for c in cls_idx:
        ap = _ap(scores[:, c], targets[:, c])
        if not np.isnan(ap):
            aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def f1_at_threshold(scores, targets, cls_idx=None, thr=0.5):
    """Per-class F1 tại threshold cố định (protocol đơn giản của RCNn eval).

    Trả dict {precision, recall, f1} trung bình trên các lớp cls_idx.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    n_cls = scores.shape[1]
    if cls_idx is None:
        cls_idx = range(n_cls)
    preds = (scores > thr).astype(np.float32)
    precs, recs, f1s = [], [], []
    for c in cls_idx:
        p = preds[:, c]
        t = targets[:, c]
        tp = float((p * t).sum())
        fp = float((p * (1 - t)).sum())
        fn = float(((1 - p) * t).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    return {
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "f1": float(np.mean(f1s)),
    }


def overall_f1(scores, targets, cls_idx=None, thr=0.5):
    """Overall F1 (micro trên toàn bộ predictions) — metric phụ."""
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if cls_idx is not None:
        scores = scores[:, list(cls_idx)]
        targets = targets[:, list(cls_idx)]
    preds = (scores > thr).astype(np.float32)
    tp = float((preds * targets).sum())
    fp = float((preds * (1 - targets)).sum())
    fn = float(((1 - preds) * targets).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def compute_topk_f1(scores, targets, k_val=3):
    """Protocol chuẩn từ RCNn / DualCoOp / SDL cho ZSL & GZSL trên MS-COCO.

    Chọn top-k nhãn có điểm cao nhất cho mỗi ảnh:
    Precision = TP / (k * N_images)
    Recall = TP / N_total_positives
    F1 = 2 * P * R / (P + R)
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.from_numpy(scores)
    if not isinstance(targets, torch.Tensor):
        targets = torch.from_numpy(targets)

    scores = scores.float()
    targets = targets.float()

    idx = scores.topk(dim=1, k=k_val)[1]
    pred_binary = torch.zeros_like(scores)
    pred_binary.scatter_(dim=1, index=idx, src=torch.ones(scores.size(0), k_val).to(scores.device))
    mask = pred_binary == 1
    TP = (targets[mask] == 1).sum().float()
    tpfp = mask.sum().float()
    tpfn = (targets == 1).sum().float()
    p = (TP / tpfp).item() if tpfp > 0 else 0.0
    r = (TP / tpfn).item() if tpfn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p * 100, "recall": r * 100, "f1": f1 * 100}
