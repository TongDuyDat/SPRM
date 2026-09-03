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
        if isinstance(refiner_cfg, dict):
            enabled = refiner_cfg.get("enabled", True)
        else:
            enabled = getattr(refiner_cfg, "enabled", True)
        self.enabled = bool(enabled)

        if self.visual_dim == self.prototype_dim:
            self.visual_gate = nn.Identity()
        else:
            self.visual_gate = nn.Linear(self.visual_dim, self.prototype_dim, bias=False)

        self.U = nn.Linear(self.visual_dim, self.prototype_dim, bias=False)
        self.Wr = nn.Linear(3 * self.prototype_dim, self.prototype_dim)
        # Optional conservative gate init: setting Wr.bias to a negative constant
        # (e.g. -2.0) makes sigmoid(Wr(gate_input)) start near 0.12, so refined
        # prototypes begin as a small perturbation of the semantic prototypes.
        # None (default) keeps nn.Linear's standard init -> backward compatible.
        if isinstance(refiner_cfg, dict):
            gate_bias_init = refiner_cfg.get("gate_bias_init", None)
        else:
            gate_bias_init = getattr(refiner_cfg, "gate_bias_init", None)
        if gate_bias_init is not None:
            nn.init.constant_(self.Wr.bias, float(gate_bias_init))
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


class CrossAttentionPrototypeRefiner(nn.Module):
    """v10: IPR core (Eqs. 18-21) whose refinement residual is read from the
    PATCH TOKENS via cross-attention instead of one CLS-derived vector.

    Diagnosis of the old refiner: q = U v_tilde carried no information the
    matcher did not already have (a single static vector per image), so
    training could only bend prototypes toward the seen cluster — the
    refined bank lost to the semantic bank on unseen at every checkpoint.
    Here the residual is an attention-weighted read of the N patch tokens
    (spatial evidence the pipeline never used):

        Q_c = W_qc(p_c)          [C, d]     class queries (semantic prototypes)
        Q_a = W_qa(a_j)          [n_attr,d] attribute queries (frozen CLIP
                                           embeddings of attribute names)
        K   = W_k(tokens)        [B, N, d]
        V   = W_vca(tokens)      [B, N, d]
        M_class = softmax(Q_c K^T / sqrt(d))            [B, C, N]
        M_attr  = softmax(Q_a K^T / sqrt(d))            [B, n_attr, N]
        g_c     = sigmoid(W_r([v; p_c; v*p_c]))          (Eq. 19 unchanged)
        p_hat_c = p_c + lambda_r * g_c * (M_class V)     (Eq. 21, q -> patch read)
        a_hat   = max_N(M_attr)                         [B, n_attr] (PSVMA
                                           Eq. 3 style, supervised by L_attn)

    lambda_r is a zero-init scalar (LP-FT anchor): at epoch 0 p_hat == p_sem
    bit-exactly, so the model degenerates to the R6b baseline if the new
    parameters learn nothing. Every new parameter is a shared function of
    (phi, tokens) — no per-class free parameters — so unseen classes inherit
    refinement through their phi alone.
    """

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
        if self.visual_dim != self.prototype_dim:
            raise ValueError(
                "CrossAttentionPrototypeRefiner requires visual_dim == "
                f"prototype_dim (got {self.visual_dim} vs {self.prototype_dim})."
            )
        self.token_dim = int(getattr(refiner_cfg, "token_dim", 768))

        if isinstance(refiner_cfg, dict):
            enabled = refiner_cfg.get("enabled", True)
            gate_bias_init = refiner_cfg.get("gate_bias_init", None)
            lambda_r_init = refiner_cfg.get("lambda_r_init", 0.0)
            attribute_query_path = refiner_cfg.get("attribute_query_path", None)
            attr_query_free = refiner_cfg.get("attr_query_free", False)
            n_attr = refiner_cfg.get("n_attr", None)
            cfg_mode = refiner_cfg.get("cfg_mode", "legacy")
            lambda_r_freeze = refiner_cfg.get("lambda_r_freeze", False)
        else:
            enabled = getattr(refiner_cfg, "enabled", True)
            gate_bias_init = getattr(refiner_cfg, "gate_bias_init", None)
            lambda_r_init = getattr(refiner_cfg, "lambda_r_init", 0.0)
            attribute_query_path = getattr(refiner_cfg, "attribute_query_path", None)
            attr_query_free = getattr(refiner_cfg, "attr_query_free", False)
            n_attr = getattr(refiner_cfg, "n_attr", None)
            cfg_mode = getattr(refiner_cfg, "cfg_mode", "legacy")
            lambda_r_freeze = getattr(refiner_cfg, "lambda_r_freeze", False)
        self.enabled = bool(enabled)
        # v11 Fix 2a (IPR-v2): classifier-free residual. "cfg" switches the
        # class-branch residual from the free-form
        #     p̂ = p_sem + λ_r·gate·(M_class V)          (legacy, v10)
        # to the condition-minus-null form (Ho & Salimans, CFG):
        #     p̂ = p_sem + λ_r·( r(V) − r_null )
        # where r_null is the SAME residual functional evaluated at the
        # trainval null condition (mean tokens + mean v_tilde, cached by the
        # trainer in `null_residual`). E0 evidence (e0_v10_a0.json): the
        # image-INDEPENDENT part of the residual is the seen-margin hack
        # (V3 constant-only nearly harmless alone, but subtracting the
        # empirical mean — V4 — gains +11.7u post-hoc over V1); the
        # image-conditioned interaction is the only part that should
        # survive. λ_r changes role from residual strength to guidance scale
        # (positive small init, unclamped — the hack channel is removed by
        # construction, not by a clamp). "legacy" keeps v10 exactly.
        self.cfg_mode = str(cfg_mode).lower()
        if self.cfg_mode not in ("legacy", "cfg"):
            raise ValueError(
                "model.refiner.cfg_mode must be 'legacy' or 'cfg'."
            )
        # E2 lesson: with a LEARNABLE lambda_r the training amplifies the
        # guidance scale (0.05 -> 0.19) and the seen-margin hack returns
        # through the image-conditioned part (u 31.2 vs 48.6 without the
        # subtraction at eval). In Ho & Salimans CFG the guidance scale is
        # a FIXED hyperparameter, not a learned parameter — so freeze it
        # when asked (lambda_r_freeze=true keeps lambda_r at its init).
        self.lambda_r_freeze = bool(lambda_r_freeze)
        if self.lambda_r_freeze and self.cfg_mode != "cfg":
            raise ValueError(
                "refiner.lambda_r_freeze=true requires cfg_mode='cfg'."
            )

        d = self.prototype_dim
        # Queries from class prototypes / attribute-name embeddings.
        self.W_qc = nn.Linear(d, d, bias=False)
        self.W_qa = nn.Linear(d, d, bias=False)
        # Keys/values from the patch-token map (768-d residual stream).
        self.W_k = nn.Linear(self.token_dim, d, bias=False)
        self.W_vca = nn.Linear(self.token_dim, d, bias=False)
        # Gate, Eq. 19: sigmoid(W_r([v; p; v*p])), conservative bias init.
        self.Wr = nn.Linear(3 * d, d)
        if gate_bias_init is not None:
            nn.init.constant_(self.Wr.bias, float(gate_bias_init))
        # LP-FT anchor: residual scale starts at 0 => p_hat == p_sem.
        self.lambda_r = nn.Parameter(torch.full((), float(lambda_r_init)))
        # E2 lesson (see the cfg_mode block above): freeze the guidance
        # scale when asked — a learnable scale amplifies (0.05 -> 0.19) and
        # the seen-margin hack returns through the image-conditioned part.
        if self.lambda_r_freeze:
            self.lambda_r.requires_grad_(False)

        # Attribute query bank: frozen CLIP text embeddings of attribute names
        # (pre-trained/<ds>_attrq_b16.pt, {"embeddings": [n_attr, d]}). Fallback
        # (attr_query_free): learnable shared zero-init bank — still valid for
        # L_attn because each query row is trained only through its own
        # attribute's target, no per-class free parameter is introduced.
        self.attr_query_free = bool(attr_query_free)
        if self.attr_query_free:
            if n_attr is None:
                raise ValueError(
                    "refiner.attr_query_free=true requires refiner.n_attr."
                )
            self.attr_queries = nn.Parameter(torch.zeros(int(n_attr), d))
            self.n_attr = int(n_attr)
        else:
            if attribute_query_path is None:
                raise ValueError(
                    "CrossAttentionPrototypeRefiner needs "
                    "model.refiner.attribute_query_path (run "
                    "tools/precompute_attribute_queries.py first) or set "
                    "attr_query_free=true with n_attr."
                )
            bank = torch.load(attribute_query_path, map_location="cpu")
            emb = bank["embeddings"].detach().float()
            if emb.dim() != 2 or emb.shape[-1] != d:
                raise ValueError(
                    f"Attribute query bank {attribute_query_path} has shape "
                    f"{tuple(emb.shape)}, expected [n_attr, {d}]."
                )
            # Non-persistent: derived from config + file, not from checkpoints.
            self.register_buffer("attr_queries", emb, persistent=False)
            self.n_attr = int(emb.shape[0])

        # v11 Fix 2a (CFG): the trainval null condition for r(∅). Filled by
        # the trainer (one trainval pass before the first optimizer step).
        # Non-persistent so checkpoints stay back-compatible; the trainer
        # rebuilds it deterministically on every run.
        self.register_buffer("null_tokens", torch.zeros(1, 1, self.token_dim), persistent=False)
        self.register_buffer("null_v_tilde", torch.zeros(1, self.prototype_dim), persistent=False)
        self._null_filled = False

    def forward(self, p_sem, v_tilde, tokens=None):
        """Return (p_hat [B,C,d] unnormalized, attr_pred [B,n_attr] or None).

        p_sem: [C,d] or [B,C,d] normalized semantic prototypes.
        v_tilde: [B,d] normalized visual representation (anchored MFA output).
        tokens: [B,N,token_dim] patch-token map (last requested block level).
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
        if p_sem.dim() == 2:
            class_count = p_sem.shape[0]
            p_expand = p_sem.unsqueeze(0).expand(batch_size, class_count, -1)
        else:
            if p_sem.shape[0] != batch_size:
                raise ValueError("Batch dimension of p_sem and v_tilde must match.")
            class_count = p_sem.shape[1]
            p_expand = p_sem

        if not self.enabled:
            return p_expand, None, torch.zeros(batch_size, self.prototype_dim, device=v_tilde.device)
        if tokens is None:
            raise ValueError("CrossAttentionPrototypeRefiner requires patch tokens.")
        if tokens.dim() != 3 or tokens.shape[0] != batch_size \
                or tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"tokens must have shape [B,N,{self.token_dim}] with B={batch_size}, "
                f"got {tuple(tokens.shape)}."
            )

        scale = self.prototype_dim ** -0.5
        K = self.W_k(tokens)                                   # [B, N, d]
        V = self.W_vca(tokens)                                 # [B, N, d]

        # Attribute branch — supervised by L_attn (trains W_qa/W_k/W_vca
        # geometry even while lambda_r == 0).
        Q_a = self.W_qa(self.attr_queries)                      # [n_attr, d]
        logits_a = torch.einsum("ad,bnd->ban", Q_a, K) * scale
        M_attr = logits_a.softmax(dim=-1)                       # [B, n_attr, N]
        attr_pred = M_attr.max(dim=-1).values                   # [B, n_attr]

        # v14 Super-CLIP Local Attribute Pooling: gom đặc trưng từ các patch
        # được chú ý bởi 312 thuộc tính để tạo ra v_local biểu diễn vi mô
        # v_local: [B, d] — trung bình có trọng số của các giá trị V theo M_attr
        v_local = torch.einsum("ban,bnd->bad", M_attr, V).mean(dim=1)  # [B, d]

        # Class branch — the Eq. 21 residual source.
        if p_sem.dim() == 2:
            Q_c = self.W_qc(p_sem)                              # [C, d]
            logits_c = torch.einsum("cd,bnd->bcn", Q_c, K) * scale
        else:
            Q_c = self.W_qc(p_expand)                            # [B, C, d]
            logits_c = torch.einsum("bcd,bnd->bcn", Q_c, K) * scale
        M_class = logits_c.softmax(dim=-1)                      # [B, C, N]
        residual = M_class @ V                                 # [B, C, d]

        # Gate, Eq. 19 on [v; p; v*p].
        v_expand = v_tilde.unsqueeze(1).expand(batch_size, class_count, -1)
        gate_input = torch.cat(
            [v_expand, p_expand, v_expand * p_expand], dim=-1
        )
        gate = torch.sigmoid(self.Wr(gate_input))               # [B, C, d]

        if self.cfg_mode == "cfg":
            # v11 Fix 2a (IPR-v2): classifier-free residual. Subtract the
            # null-condition reaction r(∅) so only the image-conditioned
            # interaction survives (the seen-margin hack lived in the
            # image-independent part — E0 V3/V4 evidence). λ_r acts as a
            # guidance scale (positive small init, unclamped).
            r_img = gate * residual                             # [B, C, d]
            r_null = self._null_residual(p_sem, p_expand, v_tilde)
            p_hat = p_expand + self.lambda_r * (r_img - r_null)
            return p_hat, attr_pred, v_local

        p_hat = p_expand + self.lambda_r * gate * residual
        return p_hat, attr_pred, v_local

    @torch.no_grad()
    def set_null_condition(self, tokens_null, v_tilde_null):
        """Cache the trainval null condition for CFG mode (Fix 2a).

        tokens_null: [1, N, token_dim] — mean patch-token map over trainval.
        v_tilde_null: [1, d] — mean v_tilde over trainval.
        Called by the trainer before the first optimizer step; the module
        re-evaluates r(∅) with its CURRENT weights on every forward.
        """
        if tokens_null.dim() != 3 or tokens_null.shape[-1] != self.token_dim:
            raise ValueError(
                f"tokens_null must be [1,N,{self.token_dim}], got {tuple(tokens_null.shape)}."
            )
        if v_tilde_null.dim() != 2 or v_tilde_null.shape[-1] != self.visual_dim:
            raise ValueError(
                f"v_tilde_null must be [1,{self.visual_dim}], got {tuple(v_tilde_null.shape)}."
            )
        dev = self.W_k.weight.device
        self.null_tokens = tokens_null.to(dev).detach()
        self.null_v_tilde = v_tilde_null.to(dev).detach()
        self._null_filled = True

    def _null_residual(self, p_sem, p_expand, v_tilde):
        """r(∅) for CFG mode: class-branch residual at the cached trainval
        null condition (mean tokens + mean v_tilde over trainval).

        The trainer fills (null_tokens, null_v_tilde) once before the first
        optimizer step; this method evaluates the CURRENT module weights at
        that condition, detached — the null reaction is treated as a
        constant w.r.t. the loss (it is recomputed each call so the
        subtraction always reflects the current weights, but no gradient
        flows into it). Returns [B, C, d] matching p_expand.
        """
        if self.null_tokens is None or self.null_v_tilde is None \
                or not self._null_filled:
            # No null condition cached yet (trainer has not run the cache
            # pass): fall back to zeros => CFG == plain r_img branch.
            return torch.zeros_like(p_expand)
        with torch.no_grad():
            p_null = p_sem if p_sem.dim() == 2 else p_sem[0]
            v_null = self.null_v_tilde.to(p_null.device)
            tokens_null = self.null_tokens.to(p_null.device)
            scale = self.prototype_dim ** -0.5
            K = self.W_k(tokens_null)                            # [1, N, d]
            V = self.W_vca(tokens_null)                          # [1, N, d]
            Q_c = self.W_qc(p_null)                             # [C, d]
            logits_c = torch.einsum("cd,bnd->bcn", Q_c, K) * scale
            M_class = logits_c.softmax(dim=-1)                   # [1, C, N]
            residual = M_class @ V                               # [1, C, d]
            v_expand = v_null.unsqueeze(1).expand(1, residual.shape[1], -1)
            p_row = p_null.unsqueeze(0).expand_as(residual)
            gate_input = torch.cat(
                [v_expand, p_row, v_expand * p_row], dim=-1
            )
            gate = torch.sigmoid(self.Wr(gate_input))            # [1, C, d]
            r_null = (gate * residual).detach()                  # [1, C, d]
        return r_null.expand(p_expand.shape[0], -1, -1)


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
