import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from model.hsprn_model import HSPRN
from losses.hsprn_loss import HSPRNLoss

OK, FAIL = "[OK]", "[FAIL]"
results = []


def check(name, cond, extra=""):
    tag = OK if cond else FAIL
    results.append((bool(cond), name))
    print(f"  {tag} {name}{(' — ' + extra) if extra else ''}")
    return cond


# ---- Build model from the v8 config, pointed at the local B/32 checkpoint ----
cfg = OmegaConf.load("config/cub_clip_v8_hsprn_b16.yaml")
cfg.model.vision.backbone_type = "clip_vit_b32"
cfg.model.vision.clip_model = "./pre-trained/ViT-B-32.pt"
cfg.model.semantic.clip_model = "./pre-trained/ViT-B-32.pt"
cfg.model.loss.lambda_trans = 0.0  # L_trans needs real class-mean cache; exercised on IBM

torch.manual_seed(0)
model = HSPRN(cfg)
criterion = HSPRNLoss(cfg)
model.eval()

C, B, D = 6, 4, 512
phi = torch.nn.functional.normalize(torch.randn(C, D), dim=-1)
text_feature = {str(i): phi[i] for i in range(C)}
images = torch.randn(B, 3, 224, 224)
labels = torch.tensor([0, 1, 2, 3])

print("=" * 72)
print("T8.2  CPU forward + loss + backward (B/32 local checkpoint)")
print("=" * 72)
out = model(text_feature, images)
check("forward returns HSPRNOutput with scores", hasattr(out, "scores") and out.scores.shape == (B, C),
      f"scores {tuple(out.scores.shape)}")
check("scores finite", torch.isfinite(out.scores).all().item())
check("v_embed normalized", torch.allclose(out.v_embed.norm(dim=-1), torch.ones(B), atol=1e-4))
check("p_ref normalized", torch.allclose(out.p_ref.norm(dim=-1), torch.ones(B, C), atol=1e-4))
check("scores_sem present (for T7 fusion)", hasattr(out, "scores_sem") and out.scores_sem.shape == (B, C))

loss, loss_dict = criterion(
    out.scores, labels,
    v_tilde=out.v_embed, p_sem=out.p_sem, p_ref=out.p_ref,
    p_sem_unnorm=out.p_sem_unnorm, phi_t=phi,
)
check("loss is finite scalar", loss.dim() == 0 and torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
print("    loss_dict:", {k: round(v, 4) for k, v in loss_dict.items()})

model.zero_grad()
loss.backward()

print("\n" + "=" * 72)
print("T8.4  requires_grad / gradient audit")
print("=" * 72)
learnable = {n: p for n, p in model.named_parameters() if p.requires_grad}
frozen_bad = [n for n, p in model.named_parameters()
              if p.requires_grad and n.split(".")[0] in ("vis_encoder",) ]
check("only psc.Ws + ipr are learnable",
      sorted({n.split('.')[0] for n in learnable}) == ["ipr", "psc"],
      f"groups={sorted({n.split('.')[0] for n in learnable})}")
expected_learn = {"psc.Ws.down.weight", "psc.Ws.up.weight",
                  "ipr.U.weight", "ipr.Wr.weight", "ipr.Wr.bias"}
check("learnable set == {Ws.down,Ws.up,IPR.U,IPR.Wr(w+b)}",
      set(learnable) == expected_learn, f"n={len(learnable)}")
got_grad = [n for n, p in learnable.items() if p.grad is not None and torch.isfinite(p.grad).all()]
check("every learnable param got a finite grad", set(got_grad) == expected_learn,
      f"{len(got_grad)}/{len(expected_learn)}")
check("vision backbone fully frozen",
      all(not p.requires_grad for p in model.vis_encoder.parameters()))
check("semantic text tower fully frozen",
      all(not p.requires_grad for p in model.psc.encoder.parameters()))
check("mfa (Wv/Wv_single) fully frozen",
      all(not p.requires_grad for p in model.mfa.parameters()))

print("\n" + "=" * 72)
print("T8.3  ZSL invariants (I1/I2/I3)")
print("=" * 72)
# I1: class-agnostic scorer — msh has NO parameters and scores = <v, p_ref_c>.
n_msh = sum(p.numel() for p in model.msh.parameters())
check("I1: matching head has 0 parameters", n_msh == 0, f"n={n_msh}")
check("I1: msh.proj is Identity", isinstance(model.msh.proj, nn.Identity))
with torch.no_grad():
    manual = (out.v_embed.unsqueeze(1) * out.p_ref).sum(-1)
check("I1: scores == dot(v_tilde, p_ref_c) recomputed manually",
      torch.allclose(manual, out.scores, atol=1e-5))
check("I1: no seen_head / per-class classifier on model",
      not hasattr(model, "seen_head") and not any("seen_head" in n for n, _ in model.named_modules()))

# I2: class info enters ONLY via prototypes (no LEARNED per-class lookup).
# CLIP's frozen subword token_embedding inside the text tower is fine (it is a
# frozen vocabulary embedding, not a per-class parameter).
bad_emb = [
    n for n, m in model.named_modules()
    if isinstance(m, nn.Embedding)
    and (any(p.requires_grad for p in m.parameters()) or not n.startswith("psc.encoder."))
]
check("I2: no learnable/out-of-tower nn.Embedding (no learned per-class lookup)",
      not bad_emb, f"offenders={bad_emb or 'none'}")

# I3: seen/unseen scored by the IDENTICAL function over the union set.
# Adding 'unseen' class prototypes must not change scores of existing classes
# (no routing / partition dependence), since IPR refines per-class.
with torch.no_grad():
    C2 = C + 2
    phi2 = torch.nn.functional.normalize(torch.randn(C2, D), dim=-1)
    phi2[:C] = phi
    tf2 = {str(i): phi2[i] for i in range(C2)}
    out2 = model(tf2, images)
check("I3: adding 2 unseen classes leaves seen scores unchanged",
      torch.allclose(out2.scores[:, :C], out.scores, atol=1e-5),
      f"max|diff|={(out2.scores[:, :C] - out.scores).abs().max().item():.2e}")
check("I3: union bank scored in one call (scores width == #classes)",
      out2.scores.shape == (B, C2))

# ---- Param accounting ----
print("\n" + "=" * 72)
print("T8.1  Param accounting")
print("=" * 72)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
vis = sum(p.numel() for p in model.vis_encoder.parameters())
text = sum(p.numel() for p in model.psc.encoder.parameters())
mfa = sum(p.numel() for p in model.mfa.parameters())
print(f"  FULL MODEL total          : {total:,}  (B/32 build)")
print(f"  learnable                 : {trainable:,}")
print(f"  vis_encoder (CLIP visual) : {vis:,}  frozen")
print(f"  psc.encoder (CLIP text)   : {text:,}  frozen  <-- held in model")
print(f"  mfa (Wv+Wv_single)        : {mfa:,}  frozen (Wv 3584->512 unused for CLIP pool path)")
b16_visual = 86_192_640
b16_total = total - vis + b16_visual
print(f"  B/16 extrapolated total   : {b16_total:,}  (visual 86,192,640 + rest)")
print(f"  B/16 visual+learnable only: {b16_visual + trainable:,}  (the quoted '~87.3M')")

n_pass = sum(1 for ok, _ in results if ok)
n_fail = sum(1 for ok, _ in results if not ok)
print("\n" + "=" * 72)
print(f"SUMMARY: {n_pass} passed, {n_fail} failed")
print("=" * 72)
if n_fail:
    for ok, name in results:
        if not ok:
            print("  FAILED:", name)
    raise SystemExit(1)
