import torch
import torch.nn as nn
import torch.nn.functional as F


class SematicBank(nn.Module):
    """Backward-compatible semantic prototype bank for Eqs. (6)-(8)."""

    def __init__(self, cfg):
        super().__init__()
        semantic_cfg = cfg.model.semantic
        text_dim = getattr(semantic_cfg, "input_dim", None)
        if text_dim is None:
            text_dim = getattr(semantic_cfg, "text_dim", 768)
        shared_dim = getattr(semantic_cfg, "shared_dim", 512)
        self.Ws = nn.Linear(text_dim, shared_dim, bias=False)

    def forward(self, phi_t):
        p_sem = self.Ws(phi_t)
        p_sem_norm = F.normalize(p_sem, p=2, dim=-1)
        return p_sem_norm, p_sem


class InstancePrototypeRefiner(nn.Module):
    """Instance-aware Prototype Refiner for Eqs. (18)-(21)."""

    def __init__(self, cfg):
        super().__init__()
        semantic_cfg = cfg.model.semantic
        aggregator_cfg = cfg.model.aggregator
        refiner_cfg = getattr(cfg.model, "refiner", object())

        self.prototype_dim = getattr(semantic_cfg, "shared_dim", 512)
        self.visual_dim = getattr(
            refiner_cfg,
            "visual_dim",
            getattr(aggregator_cfg, "out_dim", self.prototype_dim),
        )
        # Diagnostic switch: when false, the refiner is bypassed and refined
        # prototypes equal the (normalized) semantic prototypes. Lets us test
        # whether instance refinement trained on seen classes hurts unseen.
        self.enabled = bool(getattr(refiner_cfg, "enabled", True))

        if self.visual_dim == self.prototype_dim:
            self.visual_gate = nn.Identity()
        else:
            self.visual_gate = nn.Linear(self.visual_dim, self.prototype_dim, bias=False)

        self.U = nn.Linear(self.visual_dim, self.prototype_dim, bias=False)
        self.Wr = nn.Linear(3 * self.prototype_dim, self.prototype_dim)
        self._gate_is_identity = isinstance(self.visual_gate, nn.Identity)

    def forward(self, p_sem: torch.Tensor, v_tilde: torch.Tensor) -> torch.Tensor:
        """Return unnormalized refined prototypes p_hat with shape [B,C,d].
            p_sem: [C,d] or [B,C,d]
            v_tilde: [B,d]
            p_sem is the semantic prototype from the semantic bank, and 
            v_tilde is the visual representation from the aggregator. The refiner computes a gate
        """
        if v_tilde.dim() != 2:
            raise ValueError("v_tilde must have shape [B,d].")
        if v_tilde.shape[-1] != self.visual_dim:
            raise ValueError(
                f"v_tilde last dim ({v_tilde.shape[-1]}) does not match "
                f"configured visual_dim ({self.visual_dim})."
            )
        if p_sem.dim() not in (2, 3):
            raise ValueError("p_sem must have shape [C,d] or [B,C,d].")
        if p_sem.shape[-1] != self.prototype_dim:
            raise ValueError(
                f"p_sem last dim ({p_sem.shape[-1]}) does not match "
                f"configured prototype_dim ({self.prototype_dim})."
            )

        batch_size = v_tilde.shape[0]
        v_gate = self.visual_gate(v_tilde)
        if not self._gate_is_identity:
            v_gate = F.normalize(v_gate, p=2, dim=-1)

        q = self.U(v_tilde)

        if p_sem.dim() == 2:
            class_count = p_sem.shape[0]
            p_expand = p_sem.unsqueeze(0).expand(batch_size, class_count, -1)
        else:
            if p_sem.shape[0] != batch_size:
                raise ValueError("Batch dimension of p_sem and v_tilde must match.")
            class_count = p_sem.shape[1]
            p_expand = p_sem

        if not self.enabled:
            return p_expand

        v_expand = v_gate.unsqueeze(1).expand(batch_size, class_count, -1)
        q_expand = q.unsqueeze(1).expand(batch_size, class_count, -1)

        gate_input = torch.cat(
            [v_expand, p_expand, v_expand * p_expand],
            dim=-1,
        )
        gate = torch.sigmoid(self.Wr(gate_input))
        return p_expand + gate * q_expand


class RefinedPrototypeBank(nn.Module):
    """Normalize refined prototypes for Eq. (22)."""

    def __init__(self, cfg):
        super().__init__()

    def forward(self, p_hat: torch.Tensor) -> torch.Tensor:
        return F.normalize(p_hat, p=2, dim=-1)


# if __name__ == "__main__":
#     import types

#     def make_cfg(shared_dim, visual_dim):
#         """Build a minimal cfg object mimicking the expected structure."""
#         cfg = types.SimpleNamespace()
#         cfg.model = types.SimpleNamespace()
#         cfg.model.semantic = types.SimpleNamespace(shared_dim=shared_dim)
#         cfg.model.aggregator = types.SimpleNamespace(out_dim=visual_dim)
#         return cfg

#     def check_unit_norm(name, tensor):
#         norms = tensor.norm(p=2, dim=-1)
#         ok = torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
#         status = "OK" if ok else "FAIL"
#         print(f"  [{status}] {name}: L2 norms ~1 -> {ok}")

#     print("=" * 70)
#     print("Case 1: visual_dim == prototype_dim (visual_gate is Identity)")
#     print("=" * 70)
#     cfg1 = make_cfg(shared_dim=512, visual_dim=512)
#     refiner1 = InstancePrototypeRefiner(cfg1)
#     bank1 = RefinedPrototypeBank(cfg1)
#     print(f"  gate_is_identity = {refiner1._gate_is_identity}")

#     B, C, d = 4, 10, 512
#     p_sem_2d = torch.randn(C, d)
#     v_tilde = torch.randn(B, d)
#     p_hat = refiner1(p_sem_2d, v_tilde)
#     p_ref = bank1(p_hat)
#     print(f"  p_sem [C,d] input -> p_hat shape {tuple(p_hat.shape)} (expect {(B, C, d)})")
#     check_unit_norm("p_ref", p_ref)

#     print()
#     print("=" * 70)
#     print("Case 2: visual_dim != prototype_dim (visual_gate is Linear)")
#     print("=" * 70)
#     cfg2 = make_cfg(shared_dim=512, visual_dim=2048)
#     refiner2 = InstancePrototypeRefiner(cfg2)
#     bank2 = RefinedPrototypeBank(cfg2)
#     print(f"  gate_is_identity = {refiner2._gate_is_identity}")

#     B, C, d_proto, d_vis = 4, 10, 512, 2048
#     p_sem_2d = torch.randn(C, d_proto)
#     v_tilde = torch.randn(B, d_vis)
#     p_hat = refiner2(p_sem_2d, v_tilde)
#     p_ref = bank2(p_hat)
#     print(f"  p_sem [C,d] input -> p_hat shape {tuple(p_hat.shape)} (expect {(B, C, d_proto)})")
#     check_unit_norm("p_ref", p_ref)

#     print()
#     print("=" * 70)
#     print("Case 3: p_sem given as [B,C,d] (per-batch prototypes) instead of [C,d]")
#     print("=" * 70)
#     p_sem_3d = torch.randn(B, C, d_proto)
#     p_hat_3d = refiner2(p_sem_3d, v_tilde)
#     p_ref_3d = bank2(p_hat_3d)
#     print(f"  p_sem [B,C,d] input -> p_hat shape {tuple(p_hat_3d.shape)} (expect {(B, C, d_proto)})")
#     check_unit_norm("p_ref_3d", p_ref_3d)

#     print()
#     print("=" * 70)
#     print("Case 4: different batch size (B=1) and class count (C=1), sanity check")
#     print("=" * 70)
#     p_sem_small = torch.randn(1, d_proto)
#     v_small = torch.randn(1, d_vis)
#     p_hat_small = refiner2(p_sem_small, v_small)
#     print(f"  B=1, C=1 -> p_hat shape {tuple(p_hat_small.shape)} (expect (1, 1, {d_proto}))")

#     print()
#     print("=" * 70)
#     print("Case 5 (expected error): v_tilde last dim mismatched with visual_dim")
#     print("=" * 70)
#     try:
#         bad_v = torch.randn(B, 4096)  # wrong last dim, should be 2048
#         refiner2(p_sem_2d, bad_v)
#         print("  [FAIL] no error was raised")
#     except ValueError as e:
#         print(f"  [OK] raised ValueError: {e}")

#     print()
#     print("=" * 70)
#     print("Case 6 (expected error): p_sem last dim mismatched with prototype_dim")
#     print("=" * 70)
#     try:
#         bad_p_sem = torch.randn(C, 999)  # wrong last dim, should be 512
#         refiner2(bad_p_sem, v_tilde)
#         print("  [FAIL] no error was raised")
#     except ValueError as e:
#         print(f"  [OK] raised ValueError: {e}")

#     print()
#     print("=" * 70)
#     print("Case 7 (expected error): v_tilde has wrong rank (3D instead of 2D)")
#     print("=" * 70)
#     try:
#         bad_rank_v = torch.randn(B, 1, d_vis)
#         refiner2(p_sem_2d, bad_rank_v)
#         print("  [FAIL] no error was raised")
#     except ValueError as e:
#         print(f"  [OK] raised ValueError: {e}")

#     print()
#     print("=" * 70)
#     print("Case 8 (expected error): p_sem rank invalid (1D)")
#     print("=" * 70)
#     try:
#         bad_rank_p = torch.randn(d_proto)
#         refiner2(bad_rank_p, v_tilde)
#         print("  [FAIL] no error was raised")
#     except ValueError as e:
#         print(f"  [OK] raised ValueError: {e}")

#     print()
#     print("=" * 70)
#     print("Case 9 (expected error): p_sem [B,C,d] batch mismatch with v_tilde batch")
#     print("=" * 70)
#     try:
#         mismatched_p_sem = torch.randn(B + 1, C, d_proto)  # wrong batch dim
#         refiner2(mismatched_p_sem, v_tilde)
#         print("  [FAIL] no error was raised")
#     except ValueError as e:
#         print(f"  [OK] raised ValueError: {e}")

#     print()
#     print("All test cases executed.")
