"""CZSL Evaluator — port y nguyên protocol từ OWCZSL/evaluator_ge.py
(WACV 2025 'Unified Framework for Open-World Compositional Zero-shot
Learning', Jayasekara et al.). KHÔNG đụng tools/evaluator.py của SPRN.

Classical framing: mỗi composition (state, object) = 1 lớp; model SPRN
đưa score (Batch, #pairs) — tương đương score_fast_model của OWCZSL.
"""

import numpy as np
import torch
from scipy.stats import hmean


class CZSLEvaluator:
    """Đánh giá CZSL theo protocol OWCZSL: closed/open mask, bias sweep,
    AUC, best Seen/Unseen/HM."""

    def __init__(self, bench):
        """bench: dict trả về từ datasets.czsl_dataset.build_czsl_bench."""
        self.dset_name = bench["dataset_name"]
        self.attr2idx = {a: i for i, a in enumerate(bench["attrs"])}
        self.obj2idx = {o: i for i, o in enumerate(bench["objs"])}
        self.pairs = torch.LongTensor(
            [(self.attr2idx[a], self.obj2idx[o]) for a, o in bench["pairs"]]
        )
        # Port OWCZSL:15 — train_pairs chuyển sang (attr_idx, obj_idx) để
        # khớp zip(attr_truth.numpy(), obj_truth.numpy()) khi so seen/unseen.
        self.train_pairs = set(
            (self.attr2idx[a], self.obj2idx[o]) for a, o in bench["train_pairs"]
        )
        self.test_pairs = set(
            (self.attr2idx[a], self.obj2idx[o]) for a, o in bench["test_pairs"]
        )

        test_pair_set = bench["test_pairs"] + bench["train_pairs"]
        test_pair_gt = bench["test_pairs"]

        test_pair_dict = dict.fromkeys(test_pair_gt, 0)
        for attr, obj in test_pair_gt:
            test_pair_dict[(attr, obj)] = 0

        masks = [1 if pair in test_pair_set else 0 for pair in bench["pairs"]]
        self.closed_mask = torch.BoolTensor(masks)
        seen_pair_set = set(bench["train_pairs"])
        mask = [1 if pair in seen_pair_set else 0 for pair in bench["pairs"]]
        self.seen_mask = torch.BoolTensor(mask)

        oracle_obj_mask = []
        for _obj in bench["objs"]:
            m = [1 if _obj == obj else 0 for attr, obj in bench["pairs"]]
            oracle_obj_mask.append(torch.BoolTensor(m))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

    def generate_predictions(self, scores, obj_truth, bias=0.0, topk=5):
        """Port OWCZSL:59-108. scores (Batch, #pairs) trên CPU tensor."""
        def get_pred_from_scores(_scores, topk):
            _, pair_pred = _scores.topk(topk, dim=1)
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = (
                self.pairs[pair_pred][:, 0].view(-1, topk),
                self.pairs[pair_pred][:, 1].view(-1, topk),
            )
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias

        results.update({"open": get_pred_from_scores(scores, topk)})
        results.update({"unbiased_open": get_pred_from_scores(orig_scores, topk)})
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({"closed": get_pred_from_scores(closed_scores, topk)})
        results.update({"unbiased_closed": get_pred_from_scores(closed_orig_scores, topk)})

        obj_truth = obj_truth.to("cpu")
        mask = self.oracle_obj_mask[obj_truth]
        oracle_obj_scores = scores.clone()
        oracle_obj_scores[~mask] = -1e10
        oracle_obj_scores_unbiased = orig_scores.clone()
        oracle_obj_scores_unbiased[~mask] = -1e10
        results.update({"object_oracle": get_pred_from_scores(oracle_obj_scores, 1)})
        results.update(
            {"object_oracle_unbiased": get_pred_from_scores(oracle_obj_scores_unbiased, 1)}
        )
        return results

    def evaluate_predictions(self, predictions, attr_truth, obj_truth, pair_truth, allpred, topk=1):
        """Port OWCZSL:169-290. allpred: dict {(attr,obj): score_tensor (N,)}
        hoặc tensor (N, #pairs) — dùng cho bias sweep."""
        attr_truth, obj_truth, pair_truth = (
            attr_truth.to("cpu"),
            obj_truth.to("cpu"),
            pair_truth.to("cpu"),
        )

        pairs = list(zip(list(attr_truth.numpy()), list(obj_truth.numpy())))
        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)
        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(unseen_ind)

        def _process(_scores):
            attr_match = (attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk])
            obj_match = (obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk])
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            return attr_match, obj_match, match, seen_match, unseen_match

        def _add_to_dict(_scores, type_name, stats):
            base = ["_attr_match", "_obj_match", "_match", "_seen_match", "_unseen_match"]
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        obj_oracle_match = (attr_truth == predictions["object_oracle"][0][:, 0]).float()
        obj_oracle_match_unbiased = (
            attr_truth == predictions["object_oracle_unbiased"][0][:, 0]
        ).float()

        stats = dict(
            obj_oracle_match=obj_oracle_match,
            obj_oracle_match_unbiased=obj_oracle_match_unbiased,
        )

        closed_scores = _process(predictions["closed"])
        unbiased_closed = _process(predictions["unbiased_closed"])
        _add_to_dict(closed_scores, "closed", stats)
        _add_to_dict(unbiased_closed, "closed_ub", stats)

        # AUC + bias sweep — port OWCZSL:221-288
        scores = predictions["scores"]
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][unseen_ind]
        max_seen_scores = (
            predictions["scores"][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[0][:, topk - 1]
        )
        unseen_score_diff = max_seen_scores - correct_scores

        unseen_matches = stats["closed_unseen_match"].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats["closed_seen_match"].mean())
        unseen_match_max = float(stats["closed_unseen_match"].mean())
        seen_accuracy, unseen_accuracy = [], []

        if isinstance(allpred, torch.Tensor) and allpred.dim() == 2:
            base_scores = allpred.to("cpu").clone()
        else:
            base_scores = torch.stack(
                [allpred[(a, o)] for a, o in zip(
                    [p[0] for p in predictions["pair_names"]],
                    [p[1] for p in predictions["pair_names"]],
                )],
                1,
            )

        for bias in biaslist:
            scores_b = base_scores.clone()
            results = self.generate_predictions(scores_b, obj_truth, bias=bias, topk=topk)
            results = _process(results["closed"])
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(unseen_accuracy)
        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis=0)
        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats["biasterm"] = float(bias_term)
        stats["best_unseen"] = np.max(unseen_accuracy)
        stats["best_seen"] = np.max(seen_accuracy)
        stats["AUC"] = area
        stats["hm_unseen"] = unseen_accuracy[idx]
        stats["hm_seen"] = seen_accuracy[idx]
        stats["best_hm"] = max_hm
        return stats
