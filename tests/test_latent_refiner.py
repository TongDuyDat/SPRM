"""LPR numerics on CPU, no weights, no images.

    python -m pytest tests/test_latent_refiner.py -v
"""

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from model.protypes import LatentPrototypeRefiner
from tools.utils import Config

D, DT, N, B, C, K = 32, 48, 9, 4, 11, 6


def make_cfg(tmp, **refiner):
    bank = os.path.join(tmp, "Z0.pt")
    torch.save({"embeddings": F.normalize(torch.randn(K, D), dim=-1)}, bank)
    r = {"token_dim": DT, "slot_path": bank, "proto_tau": 0.07,
         "proto_rho_init": 1e-3, "proto_gate_bias_init": -2.0}
    r.update(refiner)
    return Config({"model": {"semantic": {"shared_dim": D},
                             "aggregator": {"out_dim": D}, "refiner": r}})


class TestLatentRefiner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        torch.manual_seed(0)
        self.tokens = torch.randn(B, N, DT)
        self.v = F.normalize(torch.randn(B, D), dim=-1)
        self.P = F.normalize(torch.randn(C, D), dim=-1)

    def test_shapes(self):
        ref = LatentPrototypeRefiner(make_cfg(self.tmp))
        R, attr, v_local = ref(self.tokens)
        self.assertEqual((tuple(R.shape), tuple(attr.shape), tuple(v_local.shape)),
                         ((B, K, D), (B, K), (B, D)))

    def test_inactive_class_branch_is_dot_product(self):
        """Fresh model (stages 1-2): proto inactive => scores == v P^T."""
        ref = LatentPrototypeRefiner(make_cfg(self.tmp))
        self.assertFalse(bool(ref.proto.active))
        R, _, _ = ref(self.tokens)
        torch.testing.assert_close(ref.score(self.v, self.P, R), self.v @ self.P.t())

    def _active_proto(self, **kw):
        ref = LatentPrototypeRefiner(make_cfg(self.tmp, **kw))
        torch.manual_seed(3)
        for p in ref.proto.parameters():          # non-trivial gate and W_a
            p.data.normal_(0, 0.3)
        ref.proto.rho.data.fill_(0.4)
        ref.proto.active.fill_(True)
        return ref

    @staticmethod
    def _explicit_class_branch(proto, v, P, R):
        """Reference: build p_hat [B, C, d] as the v14 class branch would."""
        d = P.shape[-1]
        gin = torch.cat([v.unsqueeze(1).expand(-1, P.shape[0], -1),
                         P.unsqueeze(0).expand(v.shape[0], -1, -1),
                         v.unsqueeze(1) * P.unsqueeze(0)], dim=-1)          # [B, C, 3d]
        g = torch.sigmoid(proto.gate(gin)).squeeze(-1)                       # [B, C]
        alpha = (torch.einsum("cd,bkd->bck", proto.W_a(P), R) / proto.tau).softmax(-1)
        p_hat = P.unsqueeze(0) + proto.lam() * g.unsqueeze(-1) * torch.einsum("bck,bkd->bcd", alpha, R)
        return (v.unsqueeze(1) * F.normalize(p_hat, dim=-1)).sum(-1)

    def test_class_branch_matches_explicit(self):
        """Closed form == materialised [B, C, d] class branch (gate + per-image alpha)."""
        ref = self._active_proto()
        R, _, _ = ref(self.tokens)
        torch.testing.assert_close(ref.score(self.v, self.P, R),
                                   self._explicit_class_branch(ref.proto, self.v, self.P, R),
                                   atol=1e-5, rtol=1e-5)

    def test_class_branch_alpha_depends_on_image(self):
        """Same class, two images -> different refined scores (not a static [C, K])."""
        ref = self._active_proto()
        R, _, _ = ref(self.tokens)
        R2 = R.clone(); R2[1] = R[0]                     # image 1 gets image 0's read-out
        s1, s2 = ref.score(self.v, self.P, R), ref.score(self.v, self.P, R2)
        self.assertFalse(torch.allclose(s1[1], s2[1]))
        torch.testing.assert_close(s1[0], s2[0])

    def test_class_branch_chunk_invariant(self):
        ref = self._active_proto()
        R, _, _ = ref(self.tokens)
        full = ref.proto.scores(self.v, self.P, R, chunk=C)
        for ch in (1, 3, 5, 10 * C):
            torch.testing.assert_close(ref.proto.scores(self.v, self.P, R, chunk=ch), full,
                                       atol=1e-6, rtol=1e-6)

    def test_class_branch_inactive_is_dot_product(self):
        """ipr.proto not yet opened (stages 1-2): scores == v P^T."""
        ref = self._active_proto()
        ref.proto.active.fill_(False)
        R, _, _ = ref(self.tokens)
        torch.testing.assert_close(ref.score(self.v, self.P, R), self.v @ self.P.t())

    def test_class_branch_gradients(self):
        ref = self._active_proto()
        R, _, _ = ref(self.tokens)
        ref.score(self.v, self.P, R).logsumexp(-1).sum().backward()
        for n in ("W_a.weight", "gate.weight", "gate.bias", "rho"):
            g = dict(ref.proto.named_parameters())[n].grad
            self.assertTrue(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, n)

    def test_rho_cannot_go_negative(self):
        ref = self._active_proto()
        ref.proto.rho.data.fill_(-3.0)
        self.assertEqual(float(ref.proto.lam()), 0.0)
        R, _, _ = ref(self.tokens)
        torch.testing.assert_close(ref.score(self.v, self.P, R), self.v @ self.P.t())

    def test_slot_bank_shape_checked(self):
        bad = os.path.join(self.tmp, "bad.pt")
        torch.save({"embeddings": torch.randn(K, D + 1)}, bad)
        with self.assertRaises(ValueError):
            LatentPrototypeRefiner(make_cfg(self.tmp, slot_path=bad))


# --- fixtures for the model/trainer integration tests (CPU, no weights) -----
_D, _DT, _N, _C = 512, 768, 9, 6          # real SPRN dims; tiny batch/classes

class _FakeVisual(torch.nn.Module):
    """Images [B, 3] -> {"cls": [B, D], "tokens": [z4, z8, z12]} deterministically."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("Wc", torch.randn(3, _D, generator=g))
        self.register_buffer("Wt", torch.randn(3, _N * _DT, generator=g))

    def forward(self, x):
        cls = x @ self.Wc
        tok = (x @ self.Wt).view(x.shape[0], _N, _DT)
        return {"cls": cls, "tokens": [tok, tok * 0.5, tok * 0.25]}


def _model_cfg(tmp, epochs=6):
    """Full SPRN cfg (3-stage phased tri-loss). Refiner paths are filled by the test."""
    return Config({
        "data": {"dataset": "TOY", "batch_size": 8, "num_workers": 0},
        "model": {
            "semantic": {"backbone_type": "embedding", "input_dim": _D,
                         "shared_dim": _D, "ws_rank": 32, "ws_anchor": 0.01},
            "aggregator": {"level_dims": [_DT] * 3, "out_dim": _D},
            "refiner": {"token_dim": _DT, "proto_tau": 0.07, "proto_rho_init": 0.05,
                        "proto_gate_bias_init": -2.0},
            "loss": {"temperature": 0.07},
        },
        "training": {
            "epochs": epochs, "warmup_epochs": 0, "lr": 1e-2, "weight_decay": 1e-4,
            "eval_freq": 100,
            "save_dir": os.path.join(tmp, "ckpt"), "log_dir": os.path.join(tmp, "log"),
            "phases": {
                "1": {"epochs": 2, "loss": {"lambda_task": 0.0, "lambda_visual": 0.0,
                                            "lambda_attr": 1.0, "lambda_cls": 0.0,
                                            "lambda_rel": 0.0},
                      "train": ["ipr"], "freeze": ["mfa.W_multi", "W_local", "psc.Ws"],
                      "zero": ["mfa.W_multi", "W_local"]},
                "2": {"epochs": 4, "loss": {"lambda_task": 1.0, "lambda_visual": 2.0,
                                            "lambda_attr": 0.0, "lambda_cls": 0.1,
                                            "lambda_rel": 0.0},
                      "train": ["mfa.W_multi", "W_local"], "freeze": ["ipr", "psc.Ws"]},
                "3": {"epochs": None, "loss": {"lambda_task": 1.0, "lambda_visual": 0.0,
                                               "lambda_attr": 0.0, "lambda_cls": 0.05,
                                               "lambda_rel": 0.2},
                      "train": ["psc.Ws"], "freeze": ["ipr", "mfa.W_multi", "W_local"]},
            },
        },
    })


if not hasattr(torch.amp, "GradScaler"):   # local torch < 2.3; IBM has it
    torch.amp.GradScaler = lambda device, enabled=True: torch.cuda.amp.GradScaler(enabled=enabled)

_EVAL_KEYS = ("zsl_T1", "gzsl_u", "gzsl_s", "gzsl_H", "cal_u", "cal_s", "cal_H",
              "cal_gamma", "hub_seen", "hub_unseen")


class TestLatentModelAndTrainer(unittest.TestCase):
    """SPRN + the real TrainerBase loop (CPU, fake visual encoder)."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        from torch.utils.data import TensorDataset
        cls.tmp = tempfile.mkdtemp()
        cls.seen, cls.unseen = np.array([0, 1, 2, 3]), np.array([4, 5])
        g = torch.Generator().manual_seed(0)
        x = torch.randn(32, 3, generator=g)
        cls.train = TensorDataset(x, torch.tensor(cls.seen)[torch.arange(32) % 4])
        cls.test_s = TensorDataset(x[:8], torch.tensor(cls.seen)[torch.arange(8) % 4])
        cls.test_u = TensorDataset(x[8:12], torch.tensor(cls.unseen)[torch.arange(4) % 2])
        cls.attribute = torch.randn(6, _D, generator=g)

    def cfg(self, target_k=K, **refiner):
        cfg = _model_cfg(self.tmp)
        bank = os.path.join(self.tmp, "Z0_%d.pt" % K)
        torch.save({"embeddings": F.normalize(torch.randn(K, _D), dim=-1)}, bank)
        tgt = os.path.join(self.tmp, "A_%d.pt" % target_k)
        torch.save({"A": torch.rand(6, target_k)}, tgt)
        cfg.model.refiner.slot_path = bank
        cfg.model.refiner.slot_target_path = tgt
        for k, v in refiner.items():
            setattr(cfg.model.refiner, k, v)
        return cfg

    def build_trainer(self, cfg):
        from losses.sprn_loss import SPRNLoss
        from model.sprn_model import SPRN
        from trainer.base import TrainerBase
        model = SPRN(cfg, vis_encoder=_FakeVisual())
        tr = TrainerBase(cfg, model, SPRNLoss(cfg), self.train, self.test_s, self.test_u,
                         self.attribute, self.seen, self.unseen, device="cpu")
        tr.evaluate_calibrated = lambda: {k: 0.0 for k in _EVAL_KEYS}
        return model, tr

    def test_forward_has_no_bcd_and_scores_match_v_P(self):
        from model.sprn_model import SPRN
        m = SPRN(self.cfg(), vis_encoder=_FakeVisual())
        m.eval()
        phi = {str(c): torch.randn(_D) for c in range(6)}
        with torch.no_grad():
            o = m(phi, torch.randn(5, 3))
        self.assertEqual(tuple(o.slot_readout.shape), (5, K, _D))
        self.assertFalse(bool(m.ipr.proto.active))
        torch.testing.assert_close(o.scores, o.v_embed @ o.p_sem.t(), atol=1e-6, rtol=0)

    def test_trainer_runs_and_uses_slot_target(self):
        cfg = self.cfg()
        model, tr = self.build_trainer(cfg)
        A = torch.load(cfg.model.refiner.slot_target_path)["A"]
        before = {k: v.clone() for k, v in model.state_dict().items()}
        tr.train()
        after = model.state_dict()
        moved = {k for k in before if not torch.equal(before[k], after[k])}
        self.assertTrue(torch.equal(tr.gt_attribute.cpu(), A))
        for p in ("ipr.W_q", "ipr.W_k", "ipr.W_v", "mfa.W_multi", "W_local", "psc.Ws"):
            self.assertTrue(any(k.startswith(p) for k in moved), p)
        self.assertFalse(any(k.startswith("ipr.Z0") for k in moved))   # buffer stays fixed

    def test_class_branch_trains_only_in_stage3(self):
        """ipr.proto: frozen and inactive in stages 1-2, trained with psc.Ws in stage 3."""
        model, tr = self.build_trainer(self.cfg())
        proto = model.ipr.proto
        seen = {}
        orig = tr._apply_phase_transition

        def spy(phase):
            orig(phase)
            seen[phase] = (bool(proto.active), proto.W_a.weight.requires_grad,
                           proto.W_a.weight.detach().clone())
        tr._apply_phase_transition = spy
        tr.train()
        self.assertEqual([seen[p][:2] for p in (1, 2)], [(False, False), (False, False)])
        self.assertEqual(seen[3][:2], (True, True))
        torch.testing.assert_close(seen[2][2], seen[1][2])               # untouched in 1-2
        self.assertFalse(torch.equal(proto.W_a.weight.detach(), seen[3][2]))   # moved in 3
        self.assertTrue(bool(model.state_dict()["ipr.proto.active"]))   # saved with ckpt

    def test_slot_target_k_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.build_trainer(self.cfg(target_k=K + 1))


if __name__ == "__main__":
    unittest.main()
