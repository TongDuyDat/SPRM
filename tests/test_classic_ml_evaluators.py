import unittest
import torch
import numpy as np
from tools.evaluator import ZSLEvaluator, compute_per_class_acc
from tools.ml_evaluator import _ap, mean_average_precision, compute_topk_f1, f1_at_threshold, overall_f1


class TestClassicEvaluator(unittest.TestCase):
    def test_compute_per_class_acc_toy(self):
        # Class 0: 2 samples, 1 correct -> 50%
        # Class 1: 1 sample, 1 correct -> 100%
        # Macro acc = (0.5 + 1.0) / 2 = 75%
        y_true = torch.tensor([0, 0, 1])
        y_pred = torch.tensor([0, 1, 1])
        class_ids = np.array([0, 1])
        acc = compute_per_class_acc(y_true, y_pred, class_ids)
        self.assertAlmostEqual(acc, 0.75)

    def test_zsl_evaluator_seen_unseen_masking(self):
        seen = np.array([0, 1])
        unseen = np.array([2, 3])
        ev = ZSLEvaluator(seen, unseen)

        # 2 test samples for unseen: true classes [2, 3]
        # sample 0 scores: [10, 20, 5, 1] -> seen class 1 has 20, but in eval_zsl seen are masked to -inf!
        # so class 2 (score 5) wins among unseen [2, 3]. It matches true 2.
        # sample 1 scores: [10, 2, 1, 8] -> class 3 (score 8) wins among unseen. Matches true 3.
        scores_u = torch.tensor([
            [10.0, 20.0, 5.0, 1.0],
            [10.0, 2.0, 1.0, 8.0]
        ])
        labels_u = torch.tensor([2, 3])

        zsl_acc = ev.eval_zsl(scores_u, labels_u)
        self.assertAlmostEqual(zsl_acc, 1.0)

        # In GZSL, seen classes are NOT masked, so sample 0 will predict 1 (seen), failing unseen!
        # sample 1: class 0 has 10, class 3 has 8 -> predicts 0 (seen), failing unseen!
        scores_s = torch.tensor([
            [10.0, 1.0, 0.0, 0.0], # true 0 -> predicts 0 (correct)
            [2.0, 8.0, 1.0, 0.0]   # true 1 -> predicts 1 (correct)
        ])
        labels_s = torch.tensor([0, 1])

        u, s, H = ev.eval_gzsl(scores_s, labels_s, scores_u, labels_u)
        self.assertAlmostEqual(s, 1.0)
        self.assertAlmostEqual(u, 0.0)
        self.assertAlmostEqual(H, 0.0)

        # Calibrated GZSL:
        # seen test sample 0 scores: [10, 1, 0, 0]
        # seen test sample 1 scores: [2, 8, 1, 0]
        # If gamma = 5.0:
        # cal seen test sample 0: [5, -4, 0, 0] -> argmax 0 (correct, s=1.0)
        # cal seen test sample 1: [-3, 3, 1, 0] -> argmax 1 (correct, s=1.0)
        # cal unseen test sample 0: [5, 15, 5, 1] -> argmax 1 (still seen)
        # cal unseen test sample 1: [5, -3, 1, 8] -> argmax 3 (correct, u=0.5!)
        # So at gamma = 5.0, s = 1.0, u = 0.5 -> H = 2*1*0.5/(1.5) = 2/3 > 0!
        cal_u, cal_s, cal_H, gamma = ev.eval_gzsl_calibrated(
            scores_s, labels_s, scores_u, labels_u, gamma_range=np.array([0.0, 5.0])
        )
        self.assertGreater(cal_H, 0.0)
        self.assertAlmostEqual(cal_s, 1.0)
        self.assertAlmostEqual(cal_u, 0.5)
        self.assertAlmostEqual(cal_H, 2.0 / 3.0)
        self.assertAlmostEqual(gamma, 5.0)


class TestMLEvaluator(unittest.TestCase):
    def test_topk_f1_toy(self):
        # 2 images, 4 classes, k=2
        # Image 0: targets = [1, 0, 1, 0], scores = [0.9, 0.1, 0.8, 0.2] -> top2 are {0, 2}. TP=2, P=2/2=1, R=2/2=1
        # Image 1: targets = [0, 1, 0, 0], scores = [0.1, 0.9, 0.5, 0.2] -> top2 are {1, 2}. TP=1, P=1/2=0.5, R=1/1=1
        # Total TP = 3
        # Total predictions = 2 * 2 = 4 -> P = 3/4 = 75%
        # Total ground truth positives = 2 + 1 = 3 -> R = 3/3 = 100%
        # F1 = 2 * 0.75 * 1.0 / (0.75 + 1.0) = 1.5 / 1.75 = 6/7 = 85.714285...%
        scores = np.array([
            [0.9, 0.1, 0.8, 0.2],
            [0.1, 0.9, 0.5, 0.2]
        ])
        targets = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 0]
        ])
        res = compute_topk_f1(scores, targets, k_val=2)
        self.assertAlmostEqual(res["precision"], 75.0)
        self.assertAlmostEqual(res["recall"], 100.0)
        self.assertAlmostEqual(res["f1"], 200 * 0.75 * 1.0 / 1.75)

    def test_map_toy(self):
        # 1 class, 3 samples. Targets: [1, 0, 1]
        # Ranked scores: sample 0 (score 0.9, y=1), sample 1 (score 0.8, y=1), sample 2 (score 0.1, y=0)
        # Perfectly ranked: rank 1: P=1/1; rank 2: P=2/2; rank 3: not pos
        # AP = (1.0 + 1.0) / 2 = 1.0
        scores = np.array([[0.9], [0.8], [0.1]])
        targets = np.array([[1], [1], [0]])
        ap = _ap(scores[:, 0], targets[:, 0])
        self.assertAlmostEqual(ap, 1.0)

        # Inverted: sample 0 (score 0.1, y=1), sample 1 (score 0.8, y=0), sample 2 (score 0.9, y=0)
        # ranks:
        # rank 1: y=0
        # rank 2: y=0
        # rank 3: y=1, P = 1/3
        # AP = (1/3) / 1 = 1/3
        scores_inv = np.array([[0.1], [0.8], [0.9]])
        targets_inv = np.array([[1], [0], [0]])
        ap_inv = _ap(scores_inv[:, 0], targets_inv[:, 0])
        self.assertAlmostEqual(ap_inv, 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
