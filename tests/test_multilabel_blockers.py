"""Regression tests for the multi-label blockers (B3 phase table, B4 index space, B5 rel term)
and LPR L_attr on multi-label data.

CPU only, no COCO images: a fake visual encoder and a synthetic 65-column dataset stand in
for the real loaders.

    python -m pytest tests/test_multilabel_blockers.py -v
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from losses.sprn_loss import SPRNLoss
from model.sprn_model import SPRN
from tools.utils import Config

if not hasattr(torch.amp, "GradScaler"):   # local torch < 2.3; IBM has it
    torch.amp.GradScaler = lambda device, enabled=True: torch.cuda.amp.GradScaler(enabled=enabled)

D, DT, N, K = 512, 768, 9, 8
# The real COCO 48/17 split, copied from bench6_data/coco/anno/cls_ids.pickle on
# IBM (2026-10-05). Seen ids are NOT the first 48 eval positions, and max id is
# 79 > 64, which is exactly what made B4 an IndexError.
COCO_SEEN = [0, 1, 2, 3, 6, 7, 8, 13, 14, 17, 18, 21, 22, 23, 24, 26, 28, 29, 30, 33, 37,
             39, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 56, 59, 61, 62, 63, 64, 65,
             68, 69, 70, 72, 73, 74, 75, 79]
COCO_UNSEEN = [4, 5, 15, 16, 19, 20, 25, 27, 31, 36, 41, 43, 55, 57, 66, 71, 76]
EVAL = sorted(COCO_SEEN + COCO_UNSEEN)


class FakeVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("Wc", torch.randn(3, D, generator=g))
        self.register_buffer("Wt", torch.randn(3, N * DT, generator=g))

    def forward(self, x):
        t = (x @ self.Wt).view(x.shape[0], N, DT)
        return {"cls": x @ self.Wc, "tokens": [t, t * 0.5, t * 0.25]}


class MultiHot(Dataset):
    def __init__(self, n, cols, seed):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, generator=g)
        self.y = (torch.rand(n, cols, generator=g) < 0.1).float()
        self.y[:, 0] = 1.0                       # every image has a positive

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


DEFAULT_PHASES = {"1": {"epochs": None,
                        "loss": {"lambda_task": 1.0, "lambda_visual": 2.0},
                        "train": ["mfa.W_multi", "W_local"],
                        "freeze": ["ipr", "psc.Ws"]}}


def coco_like_cfg(tmp, epochs=30, phases=None):
    """Mirrors config/families/multilabel.yaml with a synthetic slot bank (K=8) and target."""
    bank = os.path.join(tmp, "Z.pt")
    torch.save({"embeddings": torch.nn.functional.normalize(torch.randn(K, D), dim=-1)}, bank)
    tgt = os.path.join(tmp, "A.pt")
    torch.save({"A": torch.rand(65, K)}, tgt)
    return Config({
        "data": {"dataset": "coco", "batch_size": 8, "num_workers": 0},
        "model": {
            "semantic": {"backbone_type": "embedding", "input_dim": D,
                         "shared_dim": D, "ws_rank": 32, "ws_anchor": 0.01},
            "aggregator": {"level_dims": [DT] * 3, "out_dim": D},
            "refiner": {"token_dim": DT, "slot_path": bank, "slot_target_path": tgt,
                        "proto_tau": 0.07, "proto_rho_init": 1e-3,
                        "proto_gate_bias_init": -2.0},
            "loss": {"temperature": 0.07, "asl_gamma_neg": 4.0,
                     "asl_gamma_pos": 0.0, "asl_clip": 0.05, "lambda_task": 1.0,
                     "lambda_visual": 2.0, "lambda_ws": 0.01},
        },
        "training": {"epochs": epochs, "lr": 1e-3, "weight_decay": 1e-4,
                     "warmup_epochs": 2, "eval_freq": 1000, "amp": False,
                     "save_dir": os.path.join(tmp, "ckpt"), "log_dir": os.path.join(tmp, "log"),
                     "phases": DEFAULT_PHASES if phases is None else phases},
    })


def build_trainer(cfg, seen, unseen):
    from trainer.multilabel import MultiLabelTrainer
    model = SPRN(cfg, vis_encoder=FakeVisual())
    train = MultiHot(24, len(seen), 0)
    test = MultiHot(16, len(seen) + len(unseen), 1)
    test.eval_targets = test.y.numpy()
    attribute = torch.randn(len(seen) + len(unseen), D)
    return MultiLabelTrainer(cfg=cfg, model=model, criterion=SPRNLoss(cfg),
                             train_dataset=train, test_seen_dataset=test,
                             test_unseen_dataset=test, attribute=attribute,
                             seen_classes=np.asarray(seen), unseen_classes=np.asarray(unseen),
                             device="cpu")


class TestB3PhaseTable(unittest.TestCase):
    def test_explicit_30_epoch_phases_still_validated(self):
        phases = {1: {"epochs": 8, "train": ["mfa.W_multi"]},
                  2: {"epochs": 24, "train": ["W_local"]},
                  3: {"epochs": None, "train": ["psc.Ws"]}}
        tr = build_trainer(coco_like_cfg(tempfile.mkdtemp(), phases=phases),
                           list(range(48)), list(range(48, 65)))
        self.assertEqual([tr._phases[k]["epochs"] for k in (1, 2, 3)], [8, 24, 30])

    def test_bad_phase_boundaries_still_rejected(self):
        phases = {1: {"epochs": 8}, 2: {"epochs": 40}, 3: {"epochs": None}}
        with self.assertRaises(ValueError):
            build_trainer(coco_like_cfg(tempfile.mkdtemp(), phases=phases),
                          list(range(48)), list(range(48, 65)))


class TestB4CocoIndexSpace(unittest.TestCase):
    def remap(self):
        pos = {c: j for j, c in enumerate(EVAL)}
        return [pos[c] for c in COCO_SEEN], [pos[c] for c in COCO_UNSEEN]

    def test_raw_ids_overflow_the_65_row_phi(self):
        """The pre-fix bug: indexing a 65-row phi with raw ids raises."""
        self.assertGreater(max(COCO_SEEN), 64)
        with self.assertRaises(IndexError):
            torch.randn(65, D)[torch.as_tensor(COCO_SEEN)]

    def test_remap_is_a_partition_of_eval_positions(self):
        seen, unseen = self.remap()
        self.assertEqual(sorted(seen + unseen), list(range(65)))
        self.assertEqual(len(seen), 48)
        self.assertTrue(max(seen + unseen) < 65)

    def test_seen_positions_keep_train_column_order(self):
        """Train targets are [N, 48] in sorted raw-id order; positions must match it."""
        seen, _ = self.remap()
        self.assertEqual([EVAL[p] for p in seen], COCO_SEEN)

    def test_eval_order_matches_tag_list(self):
        """Semantics/phi rows follow bench6_classes/coco_tags.txt; eval columns must too."""
        from datasets.ml_dataset import COCO_CLASSNAMES
        tags_path = Path(__file__).resolve().parents[1] / "bench6_classes" / "coco_tags.txt"
        tags = [l.strip() for l in open(tags_path) if l.strip()]
        self.assertEqual([COCO_CLASSNAMES[i] for i in EVAL], tags)


class TestB5RelWeight(unittest.TestCase):
    def test_one_epoch_runs(self):
        """Before the fix: NameError: name 'rel_w' is not defined, at the first batch."""
        seen = list(range(48))
        tr = build_trainer(coco_like_cfg(tempfile.mkdtemp(), epochs=1), seen, list(range(48, 65)))
        stats = tr.train_one_epoch(1)
        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertGreater(stats["loss_visual"], 0.0)       # l_vis is live (was 0 in old scripts)

    def test_relational_term_is_used_when_staged(self):
        phases = {"1": {"epochs": 1, "loss": {"lambda_rel": 0.2}, "train": ["psc.Ws"]}}
        seen = list(range(48))
        tr = build_trainer(coco_like_cfg(tempfile.mkdtemp(), epochs=1, phases=phases),
                           seen, list(range(48, 65)))
        stats = tr.train_one_epoch(1)
        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertIn("loss_rel", stats)


ATTR_PHASES = {"1": {"epochs": 1, "loss": {"lambda_attr": 1.0, "lambda_task": 0.0,
                                           "lambda_visual": 0.0}, "train": ["ipr"]}}


class TestLprMultilabel(unittest.TestCase):
    """L_attr on COCO-like data: target = max over positive labels' slot rows."""

    def test_max_over_positives(self):
        from trainer.multilabel import MultiLabelTrainer
        table = torch.tensor([[1.0, 0.0, 0.2], [0.0, 2.0, 0.1], [0.5, 0.5, 3.0]])
        y = torch.tensor([[1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32)
        out = MultiLabelTrainer.multihot_slot_target(y, table)
        torch.testing.assert_close(out, torch.tensor([[1.0, 2.0, 0.2], [0.5, 0.5, 3.0], [0.0, 0.0, 0.0]]))

    def lpr_trainer(self, phases=None):
        cfg = coco_like_cfg(tempfile.mkdtemp(), epochs=1, phases=phases)
        A = torch.load(cfg.model.refiner.slot_target_path)["A"]
        seen = list(range(48))
        return build_trainer(cfg, seen, list(range(48, 65))), A, seen

    def test_seen_rows_follow_train_columns(self):
        tr, A, seen = self.lpr_trainer()
        torch.testing.assert_close(tr._seen_slot_target.cpu(), A[seen])

    def test_lattr_trains_the_refiner(self):
        tr, _, _ = self.lpr_trainer(phases=ATTR_PHASES)
        before = tr.model.ipr.W_q.weight.detach().clone()
        stats = tr.train_one_epoch(1)
        self.assertGreater(stats["loss_attr"], 0.0)
        self.assertFalse(torch.equal(before, tr.model.ipr.W_q.weight.detach()))

    def test_lattr_off_by_default(self):
        tr, _, _ = self.lpr_trainer()
        stats = tr.train_one_epoch(1)
        self.assertEqual(stats["loss_attr"], 0.0)


if __name__ == "__main__":
    unittest.main()
