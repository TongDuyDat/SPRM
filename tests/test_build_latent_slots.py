"""Slot builder core: seen-only, k-means sanity, affinity shape.

    python -m pytest tests/test_build_latent_slots.py -v
"""

import json
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from tools.build_latent_slots import (
    affinity, class_sentences, collect_sentences, row_entropy,
    seen_classes_from_images, spherical_kmeans,
)


def write_images(tmp, rows):
    p = os.path.join(tmp, "images.jsonl")
    with open(p, "w") as f:
        for split, label in rows:
            f.write(json.dumps({"split": split, "label": label}) + "\n")
    return p


class TestSeenOnly(unittest.TestCase):
    def test_split_derivation(self):
        tmp = tempfile.mkdtemp()
        p = write_images(tmp, [("trainval", 0), ("trainval", 1), ("test_seen", 1),
                               ("test_unseen", 2), ("test_unseen", 3)])
        self.assertEqual(seen_classes_from_images(p), ([0, 1], [2, 3]))

    def test_overlap_is_rejected(self):
        tmp = tempfile.mkdtemp()
        p = write_images(tmp, [("trainval", 0), ("test_unseen", 0)])
        with self.assertRaises(AssertionError):
            seen_classes_from_images(p)

    def test_unseen_sentences_never_collected(self):
        sem = [{"attr_c": [f"class {c} cue {i}" for i in range(5)]} for c in range(4)]
        texts, owner = collect_sentences(sem, [0, 1], [2, 3], 64, ("attr_c",))
        self.assertEqual(set(owner), {0, 1})
        self.assertFalse(any("class 2" in t or "class 3" in t for t in texts))

    def test_sentence_cap_is_even_not_prefix(self):
        rec = {"attr_c": [f"s{i}" for i in range(100)]}
        got = class_sentences(rec, max_n=10)
        self.assertEqual(len(got), 10)
        self.assertEqual((got[0], got[-1]), ("s0", "s99"))      # covers the whole list

    def test_string_field_is_split_by_line(self):
        self.assertEqual(class_sentences({"attr_c": "a\n\nb\n"}), ["a", "b"])


class TestKmeansAffinity(unittest.TestCase):
    def blobs(self):
        g = torch.Generator().manual_seed(0)
        centers = F.normalize(torch.randn(4, 24, generator=g), dim=-1)
        X = torch.cat([F.normalize(c + 0.05 * torch.randn(30, 24, generator=g), dim=-1)
                       for c in centers])
        return X, centers

    def test_recovers_clusters(self):
        X, true = self.blobs()
        Z, _ = spherical_kmeans(X, 4, seed=1, n_init=3)
        self.assertTrue(torch.allclose(Z.norm(dim=1), torch.ones(4), atol=1e-5))
        best = (Z @ true.t()).max(1).values
        self.assertGreater(float(best.min()), 0.98)

    def test_deterministic(self):
        X, _ = self.blobs()
        a, _ = spherical_kmeans(X, 4, seed=7, n_init=2)
        b, _ = spherical_kmeans(X, 4, seed=7, n_init=2)
        torch.testing.assert_close(a, b)

    def test_k_larger_than_n_raises(self):
        with self.assertRaises(ValueError):
            spherical_kmeans(F.normalize(torch.randn(3, 8), dim=-1), 5)

    def test_affinity_zero_rows_for_unseen(self):
        X, _ = self.blobs()
        Z, _ = spherical_kmeans(X, 4, seed=0, n_init=2)
        owner = [0] * 60 + [1] * 60
        A = affinity(X, owner, Z, n_classes=4, seen_classes=[0, 1])
        self.assertEqual(tuple(A.shape), (4, 4))
        self.assertEqual(float(A[2:].abs().sum()), 0.0)
        self.assertGreater(float(A[:2].min()), -1.0)

    def test_entropy_range(self):
        flat = torch.zeros(3, 8)
        self.assertAlmostEqual(row_entropy(flat, [0, 1, 2]), 1.0, places=5)
        peaked = torch.zeros(3, 8)
        peaked[:, 0] = 50.0
        self.assertLess(row_entropy(peaked, [0, 1, 2]), 0.05)



class TestStandardizedTarget(unittest.TestCase):
    def test_near_constant_rows_become_discriminative(self):
        from tools.build_latent_slots import mean_pair_cos, standardize_target
        g = torch.Generator().manual_seed(0)
        A = 0.9 + 0.02 * torch.rand(10, 16, generator=g)     # what raw affinity looks like
        A[8:] = 0.0                                            # unseen rows
        seen = list(range(8))
        self.assertGreater(mean_pair_cos(A[seen]), 0.99)
        T = standardize_target(A, seen)
        self.assertLess(mean_pair_cos(T[seen]), 0.6)
        self.assertTrue(bool((T >= 0).all()))
        self.assertEqual(float(T[8:].abs().sum()), 0.0)        # unseen rows untouched

    def test_stats_ignore_unseen_rows(self):
        from tools.build_latent_slots import standardize_target
        A = torch.rand(6, 4)
        T1 = standardize_target(A.clone(), [0, 1, 2, 3])
        A2 = A.clone()
        A2[4:] = 100.0                                         # change unseen rows only
        T2 = standardize_target(A2, [0, 1, 2, 3])
        torch.testing.assert_close(T1[:4], T2[:4])


if __name__ == "__main__":
    unittest.main()
