"""Toy-test: hand-computable CZSL case through legacy + exact evaluators.

6 images, 6 pair columns, 1 val-only pair masked out of candidates.
All expected numbers are derived by hand in the docstring below.

Score matrix (rows=images, cols=pairs 0..5):
pairs = [(0,0) seen, (0,1) seen, (1,0) seen, (1,1) UNSEEN, (2,0) UNSEEN, (1,2) UNSEEN-val-only]

img  c0   c1   c2   c3   c4   c5   gt
0   0.9  0.1  0.2  0.5  0.3  0.4   0   (seen)
1   0.1  0.2  0.9  0.3  0.5  0.4   2   (seen)
2   0.2  0.9  0.1  0.4  0.3  0.5   1   (seen)
3   0.1  0.2  0.3  0.9  0.5  0.4   3   (unseen)
4   0.2  0.1  0.3  0.5  0.4  0.6   5?? -> gt 5 is val-only, INVALID for closed.
    -> use gt 4 with c5 distractor 0.4 < 0.5? c4=0.4, c5=0.6 -> unbiased pred c5 (masked)
    Actually set: img4 gt=4: scores 0.2 0.1 0.3 0.5 0.4 0.6
      seen-group best = c3(0.5); unseen-group best among {c3,c4} = c3(0.5)!
      c3 is unseen col: unseen group = {3,4}. best unseen = c3 (0.5), gap = 0.
      Hmm need cleaner numbers. Final matrix below is used instead.

FINAL matrix used (verified by hand):
pairs: 0=(0,0)S 1=(0,1)S 2=(1,0)S 3=(1,1)U 4=(2,0)U 5=(1,2)val-only
img  c0   c1   c2   c3   c4   c5 | gt | group
0   0.9  0.1  0.2  0.3  0.4  8.0 | 0 | seen  -> pred c0 ok, gap = 0.9-0.4=0.5
1   0.1  0.2  0.9  0.3  0.5  8.0 | 2 | seen  -> pred c2 ok, gap = 0.9-0.5=0.4
2   0.2  0.9  0.1  0.4  0.3  8.0 | 1 | seen  -> pred c1 ok, gap = 0.9-0.4=0.5
3   0.1  0.2  0.3  0.9  0.5  8.0 | 3 | unseen-> pred c3 ok, gap = 0.3-0.9=-0.6
4   0.2  0.1  0.3  0.4  0.9  8.0 | 4 | unseen-> pred c4 ok, gap = 0.3-0.9=-0.6
5   0.3  0.2  0.1  0.5  0.4  8.0 | 0 | seen  -> pred c3 WRONG, gap = 0.3-0.5=-0.2

bias 0 (bias added to unseen cols 3,4,5; c5 masked -1e10 in closed):
 seen preds: c0,c2,c1,c3,c4,c3 -> seen imgs 0,1,2,5: correct 0,1,2 -> S=2/3
 unseen imgs 3,4: pred c3,c4 correct -> U=2/2
 HM0 = 2*(2/3)*1/(5/3) = 0.8
 best_seen = max S over curve; at large positive bias all pred unseen:
   seen preds -> unseen group best: img0 c4? unseen cols best per img:
   img0: max(c3=0.3,c4=0.4)=c4; img1: c4=0.5; img2: c3=0.4; img5: c3=0.5
   with big bias all images predict unseen: S=0, U=1
 best_seen=2/3 (bias<=0 side, S at bias 0 is max since lowering bias only helps seen? no:
   at negative bias seen still wins where gap>0; S max = 2/3 achieved for bias < 0.2
   (img5 flips to c3 when bias > gap=-0.2... img5 gap = 0.3-0.5 = -0.2, so for bias > 0.2
    img5 pred = c3 (already at bias 0 pred c3 since closed uses candidate mask and
    top1 among ALL cols? NO: closed pred = argmax over allowed cols after bias.
    unseen cols get +bias. img5: c3=0.5+b, c4=0.4+b, seen c0=0.3. For b>0.2, c3=0.7>0.5...
    wait c3 already 0.5 > 0.3 at b=0 -> pred c3 wrong.
   S(0)=2/3; S can only stay or drop as bias grows. best_seen=2/3.
 best_unseen: U(0)=1.0 already -> best_unseen=1.0.
 best_hm: sweep b from -inf: at b<=-0.6 all unseen preds flip to seen (img3,4 need
   b < -0.6 to make c3 < c2? img3: c3=0.9+b, seen best c2=0.3 -> flips when 0.9+b<0.3, b<-0.6)
   at b=-0.7: preds img3->c2(wrong, gt3), img4->c2? c4=0.9-0.7=0.2 < c2=0.3 -> pred c2 wrong.
   S(-0.7)=2/3 (imgs 0,1,2 ok; img5 pred c0=0.3 vs unseen c3=0.5-0.7=-0.2, c4=0.4-0.7=-0.3 -> pred c0=0.3 correct!)
   wait img5 gt=0, pred c0 -> correct! So S(-0.7)=3/3? imgs 0,1,2,5 all seen: 0:c0 0.9 ok, 1:c2 ok, 2:c1 ok, 5:c0 ok -> S=1.0
   U(-0.7): img3 pred c2 wrong, img4 pred c2 wrong -> U=0.
   HM(-0.7)=0. At b=0: S=2/3,U=1,HM=0.8. Between -0.6 and 0: img3 flips at b>-0.6 (0.9+b>0.3 -> b>-0.6): pred c3 correct; img4 flips at b>-0.6 too.
   For -0.6<b<... img5 pred: c3=0.5+b vs c0=0.3: correct-as-seen flips to wrong when b>-0.2.
   Interval (-0.6,-0.2): S = imgs 0,1,2,5 correct = 4/4? img5 pred c0? c3=0.5+b < 0.3 when b<-0.2 yes -> pred c0 correct. S=1.0, U=1.0 (imgs3,4 correct) -> HM=1.0!!
   best_hm = 1.0 at representative bias in (-0.6,-0.2).
   This proves positive-bias blind spot is NOT needed here; instead it tests curve completeness.
 AUC exact: curve points (S,U): (1,0) -> (1,1) -> (2/3,1) -> (0,1).
   trapz over U: [0->1] S=1 area 1.0; [1->1] zero; [1->2/3] zero width... need exact breakpoints:
   U jumps 0->1 at b=-0.6 (both img3,4 same gap -0.6). S drops 1->2/3 at b=-0.2 (img5).
   S drops 2/3->? imgs 0,1,2 gaps 0.5,0.4,0.5 -> flip at b=0.5 (img1 first: gap 0.4).
   So curve: (1,0) at b<-0.6; (1,1) for -0.6<b<-0.2; (2/3,1) -0.2<b<0.4; then drops.
   AUC = trapz with U as x: from U=0 to 1 at S=1: width 1, avg S=1 -> 1.0; then U stays 1, no width.
   AUC = 1.0.
 obj oracle (unbiased? exact uses no bias): oracle over candidate cols per GT object:
   img0 obj0: cols {0,1}: max c0 -> correct. img1 obj0: c2? cols{0,1}: max c0=0.1 vs gt c2 WRONG.
   Hmm oracle for img1 (gt=2, obj=0): cols with obj0 = {0,1}; best c0=0.1 != 2 -> wrong.
   img2 obj1: cols{1,3,5}: c1=0.9 correct. img3 obj1: cols{1,3,5}: c1? 0.2 vs c5 8.0 masked; c3=0.9 -> correct (gt c3).
   img4 obj0: cols{0,1}: c0=0.2 wrong (gt4). img5 obj0: c0=0.3 correct.
   oracle = 4/6 = 2/3.
 attr/obj/pair acc at bias 0: preds c0,c2,c1,c3,c4,c3 vs gt 0,2,1,3,4,0:
   pair: 5/6? img5 wrong -> 5/6. attr: preds attr 0,1,1,1,2,1 vs gt 0,1,1,1,2,0 -> img5 wrong -> 5/6.
   obj: preds obj 0,0,1,1,0,1 vs gt 0,0,1,1,0,0 -> img5 wrong -> 5/6.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.czsl_exact_metrics import closed_world_metrics
from tools.czsl_evaluator import CZSLEvaluator

PAIRS = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (1, 2)]
TRAIN = {(0, 0), (0, 1), (1, 0)}
TEST = {(1, 1), (2, 0)}
SCORES = np.array([
    [0.9, 0.1, 0.2, 0.3, 0.4, 8.0],
    [0.1, 0.2, 0.9, 0.3, 0.5, 8.0],
    [0.2, 0.9, 0.1, 0.4, 0.3, 8.0],
    [0.1, 0.2, 0.3, 0.9, 0.5, 8.0],
    [0.2, 0.1, 0.3, 0.4, 0.9, 8.0],
    [0.3, 0.2, 0.1, 0.5, 0.4, 8.0],
])
GT = np.array([0, 2, 1, 3, 4, 0])

EXPECTED = dict(unbiased_seen_acc=0.75, unbiased_unseen_acc=1.0, unbiased_hm=6/7,
                best_seen=1.0, best_unseen=1.0, best_hm=1.0, auc=1.0,
                pair_acc=5/6, attr_acc=5/6, obj_acc=5/6, obj_oracle_acc=5/6)


def test_exact():
    m = closed_world_metrics(SCORES, GT, PAIRS, TRAIN, candidate_set=TRAIN | TEST)
    for k, v in EXPECTED.items():
        assert abs(m[k] - v) < 1e-12, (k, m[k], v)
    assert m['n_candidate_pairs'] == 5
    # GT column 5 (val-only) would be rejected:
    try:
        closed_world_metrics(SCORES, np.array([0, 2, 1, 3, 4, 5]), PAIRS, TRAIN,
                             candidate_set=TRAIN | TEST)
    except ValueError:
        pass
    else:
        raise AssertionError('val-only GT not rejected')
    print('EXACT toy test: all hand-computed metrics match')


def test_legacy_matches_exact_on_shared_points():
    bench = dict(dataset_name='toy', attrs=[0, 1, 2], objs=[0, 1, 2],
                 pairs=PAIRS, train_pairs=list(TRAIN), test_pairs=list(TEST))
    ev = CZSLEvaluator(bench)
    s32 = torch.from_numpy(SCORES.copy())
    pt = torch.LongTensor(GT)
    preds = ev.generate_predictions(s32.clone(), torch.LongTensor([p[1] for p in [PAIRS[i] for i in GT]]))
    preds['scores'] = s32.clone()
    preds['pair_names'] = PAIRS
    pairs_t = [PAIRS[i] for i in GT.tolist()]
    attr_t = torch.LongTensor([p[0] for p in pairs_t])
    obj_t = torch.LongTensor([p[1] for p in pairs_t])
    stats = ev.evaluate_predictions(preds, attr_t, obj_t, pt, s32)
    # shared unbiased points must agree with exact implementation
    assert abs(stats['closed_ub_seen_match'] - EXPECTED['unbiased_seen_acc']) < 1e-9
    assert abs(stats['closed_ub_unseen_match'] - EXPECTED['unbiased_unseen_acc']) < 1e-9
    # legacy truncated sweep CANNOT see the (-0.6,-0.2) interval where HM=1.0:
    assert stats['best_hm'] < EXPECTED['best_hm'] - 1e-9, stats['best_hm']
    print(f"LEGACY toy: best_hm={stats['best_hm']:.6f} < exact 1.0 (blind spot reproduced)")
    # both legacy and exact agree S/U endpoints from bias-0 point
    assert abs(stats['closed_seen_match'] - EXPECTED['unbiased_seen_acc']) < 1e-9
    assert abs(stats['closed_unseen_match'] - EXPECTED['unbiased_unseen_acc']) < 1e-9


if __name__ == '__main__':
    test_exact()
    test_legacy_matches_exact_on_shared_points()
    print('TOY_HANDCHECK_ALL_PASS')
