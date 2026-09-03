import warnings

warnings.filterwarnings("ignore")
import types

import torch

from model.protypes import InstancePrototypeRefiner


def make_cfg(gbi=None):
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.semantic = types.SimpleNamespace(shared_dim=512)
    cfg.model.aggregator = types.SimpleNamespace(out_dim=512)
    ref = {"enabled": True}
    if gbi is not None:
        ref["gate_bias_init"] = gbi
    cfg.model.refiner = ref
    return cfg


def tag(ok):
    return "OK" if ok else "FAIL"


torch.manual_seed(0)
B, C, d = 2, 5, 512
p_sem = torch.randn(C, d)
v = torch.randn(B, d)
p_expand = p_sem.unsqueeze(0).expand(B, C, -1)

# 1) backward compat: no gate_bias_init -> default nn.Linear init (bias ~0, NOT -2)
r_def = InstancePrototypeRefiner(make_cfg())
b = r_def.Wr.bias
ok1 = abs(b.mean().item()) < 0.05 and not torch.allclose(b, torch.full_like(b, -2.0))
print(f"[{tag(ok1)}] default keeps nn.Linear init (bias mean={b.mean().item():.4f})")

# 2) gate_bias_init=-2.0 -> bias exactly -2
r_g = InstancePrototypeRefiner(make_cfg(gbi=-2.0))
ok2 = torch.allclose(r_g.Wr.bias, torch.full_like(r_g.Wr.bias, -2.0))
print(f"[{tag(ok2)}] gate_bias_init=-2.0 sets Wr.bias to -2.0 (bias[0]={r_g.Wr.bias[0].item():.4f})")

# 3) gate ~ sigmoid(-2) ~ 0.12 -> refined is a small perturbation of p_sem
with torch.no_grad():
    p_hat_def = r_def(p_sem, v)
    p_hat_g = r_g(p_sem, v)
d_def = (p_hat_def - p_expand).norm(dim=-1).mean().item()
d_g = (p_hat_g - p_expand).norm(dim=-1).mean().item()
print(f"    mean ||gate*q|| default={d_def:.4f}  gate_bias=-2={d_g:.4f}  (expect -2 case smaller)")
print(f"[{tag(d_g < d_def)}] conservative init reduces initial refinement magnitude")

# 4) param count unchanged + autograd flows
n_def = sum(p.numel() for p in r_def.parameters())
n_g = sum(p.numel() for p in r_g.parameters())
print(f"[{tag(n_def == n_g)}] param count unchanged ({n_g:,})")
out = r_g(p_sem, v)
out.sum().backward()
ok4 = r_g.Wr.bias.grad is not None and torch.isfinite(r_g.Wr.bias.grad).all()
print(f"[{tag(ok4)}] autograd flows (Wr.bias grad finite)")

all_ok = ok1 and ok2 and (d_g < d_def) and (n_def == n_g) and ok4
print("T5 RESULT:", "PASS" if all_ok else "FAIL")
raise SystemExit(0 if all_ok else 1)
