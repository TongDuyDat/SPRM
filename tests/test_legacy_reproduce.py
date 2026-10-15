"""Tests reproducing limitations and bugs of legacy CZSLEvaluator.
Demonstrates:
1. bias=0 point appended at end of curve with hardcoded biasterm=1000 label if chosen.
2. unseen samples that are incorrect at bias=0 never get sweep thresholds generated for them.
3. in-place mutation of input scores tensor in generate_predictions.
4. closed_mask assumption and closed vs open candidates.
"""
import unittest
import torch
import numpy as np
from tools.czsl_evaluator import CZSLEvaluator


def build_toy_bench():
    attrs = ["red", "blue"]
    objs = ["apple", "banana"]
    pairs = [
        ("red", "apple"),      # pair 0: seen
        ("red", "banana"),     # pair 1: unseen
        ("blue", "apple"),     # pair 2: unseen
        ("blue", "banana"),    # pair 3: seen
    ]
    train_pairs = [("red", "apple"), ("blue", "banana")]
    test_pairs = [("red", "banana"), ("blue", "apple"), ("red", "apple")]
    return {
        "dataset_name": "toy",
        "attrs": attrs,
        "objs": objs,
        "pairs": pairs,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
    }


class TestLegacyEvaluatorBugs(unittest.TestCase):
    def test_legacy_input_score_mutation(self):
        """Legacy generate_predictions mutates `scores` in place!"""
        bench = build_toy_bench()
        ev = CZSLEvaluator(bench)
        scores = torch.zeros(2, 4)
        scores_copy = scores.clone()
        obj_truth = torch.LongTensor([0, 1])
        ev.generate_predictions(scores, obj_truth, bias=5.0, topk=1)
        # In legacy CZSLEvaluator: scores[~mask] += bias modifies scores in-place
        self.assertFalse(torch.allclose(scores, scores_copy), "Legacy evaluator unexpectedly mutated scores in-place")

    def test_legacy_unseen_only_correct_bias_generation(self):
        """Legacy evaluator only collects bias candidate thresholds from unseen samples
        that were ALREADY CORRECT at bias=0! If an unseen sample was beaten by a seen class
        at bias=0, it is in stats['closed_unseen_match'] == 0, so line 149 drops it:
            unseen_matches = stats["closed_unseen_match"].bool()
            correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4
        This means unseen samples that need positive bias to be recovered NEVER contribute to biaslist!
        """
        bench = build_toy_bench()
        ev = CZSLEvaluator(bench)
        # 1 test sample: unseen pair ('red', 'banana') idx 1.
        # At bias 0: seen pair ('red', 'apple') idx 0 score=10.0,
        #            unseen pair ('red', 'banana') idx 1 score=8.0.
        # So at bias 0, prediction is seen (0), unseen is incorrect!
        # At bias >= 2.0, unseen pair gets score >= 10.0 and would become correct.
        # But legacy evaluator filters: unseen_score_diff[unseen_matches]!
        # Since unseen_match is False at bias 0, correct_unseen_score_diff is EMPTY!
        scores = torch.tensor([[10.0, 8.0, 0.0, 0.0]])
        pair_truth = torch.tensor([1])  # unseen
        attr_truth = torch.tensor([0])
        obj_truth = torch.tensor([1])

        preds = ev.generate_predictions(scores.clone(), obj_truth, topk=1)
        preds["scores"] = scores.clone()
        preds["pair_names"] = bench["pairs"]

        stats = ev.evaluate_predictions(preds, attr_truth, obj_truth, pair_truth, scores.clone())
        # Because correct_unseen_score_diff was empty, biaslist is empty!
        # The only point evaluated was the appended (seen_match_max, unseen_match_max).
        # And AUC is 0, best_unseen is 0, even though bias=2.1 would recover 100% unseen!
        self.assertEqual(stats["best_unseen"], 0.0)
        self.assertEqual(stats["AUC"], 0.0)

    def test_legacy_bias_sweep_endpoints_and_label(self):
        """Legacy evaluator appends (seen_match_max, unseen_match_max) at the end of the curve.
        If the max HM occurs at this point, it sets biasterm = 1000, even though this point
        was evaluated at bias=0!
        """
        bench = build_toy_bench()
        ev = CZSLEvaluator(bench)
        # Both seen and unseen are correct at bias=0
        scores = torch.tensor([
            [10.0, 0.0, 0.0, 0.0],  # seen pair 0 ('red', 'apple') correct
            [0.0, 10.0, 0.0, 0.0],  # unseen pair 1 ('red', 'banana') correct
        ])
        pair_truth = torch.tensor([0, 1])
        attr_truth = torch.tensor([0, 0])
        obj_truth = torch.tensor([0, 1])

        preds = ev.generate_predictions(scores.clone(), obj_truth, topk=1)
        preds["scores"] = scores.clone()
        preds["pair_names"] = bench["pairs"]

        stats = ev.evaluate_predictions(preds, attr_truth, obj_truth, pair_truth, scores.clone(), topk=1)
        # At bias=0, seen=1.0, unseen=1.0. Max HM is at bias=0 (seen=1.0, unseen=1.0, HM=1.0).
        # But idx == len(biaslist) which legacy labels as bias_term = 1000.0!
        self.assertEqual(stats["biasterm"], 1000.0)
        self.assertEqual(stats["best_hm"], 1.0)


if __name__ == "__main__":
    unittest.main()
