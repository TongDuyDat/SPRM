"""Tiny CPU tests for a complete CZSL bias curve, not the legacy sweep."""
import unittest
import numpy as np
from tools.czsl_exact_metrics import closed_world_metrics


class CompleteCurveTest(unittest.TestCase):
    def test_positive_bias_recovers_unseen_and_counts_images(self):
        # At bias zero no unseen prediction is correct; at bias 1.5 all are.
        pairs = [('a', 'x'), ('b', 'x')]
        scores = np.array([[3., 0.], [2., 1.]], dtype=np.float32)
        original = scores.copy()
        result = closed_world_metrics(scores, np.array([0, 1]), pairs, {pairs[0]})
        self.assertEqual(result['best_unseen'], 1.)
        self.assertEqual(result['best_hm'], 1.)
        self.assertEqual(result['n_seen_images'], 1)
        self.assertEqual(result['n_unseen_images'], 1)
        np.testing.assert_array_equal(scores, original)

    def test_curve_matches_bruteforce_and_masks_distractor(self):
        pairs = [('a','x'), ('b','x'), ('c','y'), ('d','y')]
        scores = np.random.default_rng(42).normal(size=(30,4))
        gt = np.tile([0,1,2],10)
        allowed = set(pairs[:3])
        seen = {pairs[0]}
        result = closed_world_metrics(scores,gt,pairs,seen,allowed)
        for bias, s, u in zip(result['curve']['bias'], result['curve']['seen'],
                              result['curve']['unseen']):
            adjusted = scores.copy()
            adjusted[:,1:] += bias
            adjusted[:,3] = -np.inf
            correct = adjusted.argmax(1) == gt
            self.assertAlmostEqual(s,correct[gt==0].mean())
            self.assertAlmostEqual(u,correct[gt!=0].mean())
        self.assertAlmostEqual(result['auc'], np.trapz(result['curve']['seen'],result['curve']['unseen']))
        maximum = max(2*s*u/(s+u) if s+u else 0 for s,u in zip(result['curve']['seen'],result['curve']['unseen']))
        self.assertEqual(result['best_hm'],maximum)
        self.assertEqual(result['n_candidate_pairs'],3)

    def test_rejects_masked_ground_truth(self):
        pairs=[('a','x'),('b','x')]
        with self.assertRaises(ValueError):
            closed_world_metrics(np.ones((2,2)),np.array([0,1]),pairs,{pairs[0]},{pairs[0]})


if __name__ == '__main__':
    unittest.main()
