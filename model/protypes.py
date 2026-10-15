import torch
import torch.nn as nn
import torch.utils.checkpoint


class _ProtoPath(nn.Module):
    """Instance-aware class path routed through the slots so that no [B, C, d]
    tensor exists.

    A class attends to the K slot read-outs of the same image (class -> slot -> patch). R_b already summarises the patches per slot:
        alpha_bck = softmax_k( <W_a p_c, R_bk> / tau )                  [B, C, K]
        g_bc      = sigmoid( w_v.v_b + w_p.p_c + <w_x (.) v_b, p_c> + b ) [B, C]
        p_hat_bc  = p_c + lam * g_bc * sum_k alpha_bck R_bk              (never built)
    g is the v14 gate sigmoid(W_r[v; p; v*p]) with a scalar output: a vector
    gate under a sigmoid has no closed form, so it would bring [B, C, d] back.

    Scores and ||p_hat|| come from the closed form, per block of classes:
        <v, p_hat>   = <v, p> + sum_k coef_bck (R_bk . v_b)
        ||p_hat||^2  = ||p||^2 + 2 sum_k coef_bck (R_bk . p_c) + coef^T (R R^T) coef
    with coef = lam * g * alpha. The largest tensor is [B, chunk, K].

    ``active`` (a buffer, saved with the checkpoint) is False until the stage
    that trains this path, so earlier stages score exactly v P^T. lam = clamp(rho, 0):
    rho starts at 1e-3 because clamp has zero gradient at 0.
    """

    def __init__(self, d, tau, rho_init, gate_bias_init=-2.0):
        super().__init__()
        self.W_a = nn.Linear(d, d, bias=False)
        self.tau = tau
        self.rho = nn.Parameter(torch.full((), rho_init))
        self.gate = nn.Linear(3 * d, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias_init))
        self.register_buffer("active", torch.zeros((), dtype=torch.bool))

    def lam(self):
        return self.rho.clamp(min=0.0)

    def _block(self, v, Pc, R, Rv, G, lam):
        d = Pc.shape[-1]
        w_v, w_p, w_x = self.gate.weight[0, :d], self.gate.weight[0, d:2 * d], self.gate.weight[0, 2 * d:]
        g = torch.sigmoid((v @ w_v).unsqueeze(1) + (Pc @ w_p).unsqueeze(0)
                          + (v * w_x) @ Pc.t() + self.gate.bias)           # [B, c]
        alpha = (torch.einsum("cd,bkd->bck", self.W_a(Pc), R) / self.tau).softmax(-1)  # [B, c, K]
        coef = (lam * g).unsqueeze(-1) * alpha                             # [B, c, K]
        num = v @ Pc.t() + torch.einsum("bck,bk->bc", coef, Rv)
        RP = torch.einsum("bkd,cd->bck", R, Pc)                            # [B, c, K]
        sq = ((Pc * Pc).sum(-1).unsqueeze(0) + 2 * (coef * RP).sum(-1)
              + torch.einsum("bck,bkl,bcl->bc", coef, G, coef))
        return num / sq.clamp_min(1e-12).sqrt()

    def scores(self, v, P, R, chunk=16384):
        """Cosine scores [B, C] against the per-image refined prototypes."""
        Rv = torch.einsum("bkd,bd->bk", R, v)                               # [B, K]
        G = torch.einsum("bkd,bld->bkl", R, R)                              # [B, K, K]
        lam = self.lam()
        use_ckpt = torch.is_grad_enabled() and P.shape[0] > chunk
        out = []
        for i in range(0, P.shape[0], chunk):
            args = (v, P[i:i + chunk], R, Rv, G, lam)
            out.append(torch.utils.checkpoint.checkpoint(self._block, *args, use_reentrant=False)
                       if use_ckpt else self._block(*args))
        return torch.cat(out, dim=1)


class LatentPrototypeRefiner(nn.Module):
    """LPR: K latent slots read the patch tokens.

    Local branch:
        M       = softmax((Z W_q)(T W_k)^T / sqrt(d))      [B, K, N]
        R       = M (T W_v)                                [B, K, d]
        attr    = max_N M                                  [B, K]   -> L_attr
        v_local = mean_K R                                 [B, d]   -> fusion
    Class branch (``proto``, see _ProtoPath): per-image refined prototypes scored in
    closed form from R; the [B, C, d] tensor never exists.
    """

    def __init__(self, cfg):
        super().__init__()
        d = self.prototype_dim = cfg.model.semantic.shared_dim
        rc = cfg.model.refiner
        self.token_dim = int(rc.token_dim)
        slot_path = rc.slot_path
        Z0 = torch.load(slot_path, map_location="cpu")["embeddings"].detach().float()
        if Z0.dim() != 2 or Z0.shape[-1] != d:
            raise ValueError(f"slot bank {slot_path} has shape {tuple(Z0.shape)}, "
                             f"expected [K, {d}].")
        self.n_slots = int(Z0.shape[0])
        self.register_buffer("Z0", Z0)

        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(self.token_dim, d, bias=False)
        self.W_v = nn.Linear(self.token_dim, d, bias=False)
        self.proto = _ProtoPath(d, float(rc.proto_tau), float(rc.proto_rho_init),
                                float(rc.proto_gate_bias_init))

    def forward(self, tokens):
        """Return (R [B,K,d], attr_pred [B,K], v_local [B,d])."""
        if tokens.dim() != 3 or tokens.shape[-1] != self.token_dim:
            raise ValueError(f"tokens must be [B,N,{self.token_dim}], got {tuple(tokens.shape)}.")
        scale = self.prototype_dim ** -0.5
        Q = self.W_q(self.Z0)                                   # [K, d]
        K = self.W_k(tokens)                                    # [B, N, d]
        V = self.W_v(tokens)                                    # [B, N, d]
        M = (torch.einsum("kd,bnd->bkn", Q, K) * scale).softmax(dim=-1)   # [B, K, N]
        R = M @ V                                               # [B, K, d]
        return R, M.max(dim=-1).values, R.mean(dim=1)

    def score(self, v, P, R):
        """Cosine scores [B, C] of v against the refined prototypes of P."""
        if not bool(self.proto.active):
            return v @ P.t()
        return self.proto.scores(v, P, R)
