"""Complete Top-1 CZSL bias curve from saved scores (fractions, not percent).

Adds one common bias to all unseen candidates. Exhaustively evaluates every
reachable interval between changes of the best seen/unseen group. Equal score
within a group chooses the lowest column; equal seen/unseen score chooses seen.
Exact curve (all score-gap breakpoints) is distinct from published 20-bin AUC.
No feasibility filtering is introduced. Input arrays are never modified.
"""
import numpy as np


def closed_world_metrics(scores, pair_labels, pairs, train_pair_set, candidate_set=None):
    scores = np.asarray(scores)
    gt = np.asarray(pair_labels)
    n, c = scores.shape
    if n == 0 or len(pairs) != c or gt.shape != (n,):
        raise ValueError('Invalid score/label/pair dimensions')
    if not np.issubdtype(gt.dtype, np.integer) or gt.min() < 0 or gt.max() >= c:
        raise ValueError('Labels must be valid integer column indices')
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite scores')
    candidates = np.array([candidate_set is None or p in candidate_set for p in pairs])
    seen = np.array([p in train_pair_set for p in pairs])
    if not candidates[gt].all():
        raise ValueError('Ground truth is excluded by candidate mask')
    sc = np.flatnonzero(candidates & seen)
    uc = np.flatnonzero(candidates & ~seen)
    seen_img = seen[gt]
    ns, nu = int(seen_img.sum()), int((~seen_img).sum())
    if not len(sc) or not len(uc) or not ns or not nu:
        raise ValueError('Both seen/unseen candidate and image groups are required')

    # Only two column subsets are materialized; never clone NxC per bias.
    ps = sc[np.argmax(scores[:, sc], axis=1)]
    pu = uc[np.argmax(scores[:, uc], axis=1)]
    row = np.arange(n)
    gap = scores[row, ps].astype(np.float64) - scores[row, pu].astype(np.float64)
    correct_s = (ps == gt) & seen_img
    correct_u = (pu == gt) & ~seen_img
    thresholds, group = np.unique(gap, return_inverse=True)
    losses = np.bincount(group, weights=correct_s, minlength=len(thresholds))
    gains = np.bincount(group, weights=correct_u, minlength=len(thresholds))
    s = np.r_[correct_s.sum(), correct_s.sum() - np.cumsum(losses)] / ns
    u = np.r_[0., np.cumsum(gains)] / nu
    hm = np.divide(2*s*u, s+u, out=np.zeros_like(s), where=(s+u)>0)
    # Representatives for each interval, including fully seen/unseen endpoints.
    middle = thresholds[:-1] + (thresholds[1:] - thresholds[:-1])/2
    biases = np.r_[thresholds[0]-1., middle, thresholds[-1]+1.]
    best = int(np.argmax(hm))
    pred = np.where(gap >= 0, ps, pu)
    correct = pred == gt
    s0, u0 = float(correct[seen_img].mean()), float(correct[~seen_img].mean())
    attrs = np.array([p[0] for p in pairs])
    objs = np.array([p[1] for p in pairs])
    oracle_correct = 0
    for obj in np.unique(objs[gt]):
        rows = np.flatnonzero(objs[gt] == obj)
        cols = np.flatnonzero(candidates & (objs == obj))
        prediction = cols[np.argmax(scores[np.ix_(rows, cols)], axis=1)]
        oracle_correct += int((prediction == gt[rows]).sum())
    integrate = getattr(np, 'trapezoid', None) or np.trapz
    return {
        'protocol': 'exact_top1_gap_sweep_v1',
        'averaging': 'per-image', 'units': 'fraction',
        'tie_rule': 'within-group lowest column; cross-group seen wins equality',
        'n_images': n, 'n_score_columns': c, 'n_candidate_pairs': int(candidates.sum()),
        'n_seen_images': ns, 'n_unseen_images': nu,
        'n_seen_candidates': len(sc), 'n_unseen_candidates': len(uc),
        'unbiased_seen_acc': s0, 'unbiased_unseen_acc': u0,
        'unbiased_hm': 2*s0*u0/(s0+u0) if s0+u0 else 0.,
        'pair_acc': float(correct.mean()),
        'attr_acc': float((attrs[pred] == attrs[gt]).mean()),
        'obj_acc': float((objs[pred] == objs[gt]).mean()),
        'obj_oracle_acc': oracle_correct/n,
        'best_seen': float(s.max()), 'best_unseen': float(u.max()),
        'best_hm': float(hm[best]), 'hm_seen': float(s[best]), 'hm_unseen': float(u[best]),
        'biasterm': float(biases[best]), 'auc': float(integrate(s, u)),
        'curve': {'bias': biases.tolist(), 'seen': s.tolist(), 'unseen': u.tolist()},
    }
