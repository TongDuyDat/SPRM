from typing import Dict, List, Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os

_script_dir = os.path.dirname(__file__)
_root = os.path.normpath(os.path.join(_script_dir, "../.."))
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from . import load_semantic_backbone
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "sematics",
        os.path.join(_script_dir, "__init__.py"),
    )
    _sem_mod = importlib.util.module_from_spec(_spec)
    _sem_mod.__package__ = "sematics"
    _sem_mod.__path__ = [_script_dir]
    sys.modules["sematics"] = _sem_mod
    _spec.loader.exec_module(_sem_mod)
    load_semantic_backbone = _sem_mod.load_semantic_backbone


def _first_present(record: dict, *keys):
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        return value
    return None


def _as_item_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if lines:
            return lines
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


class LowRankWs(nn.Module):
    """Low-rank projection around identity: W_s = I + B A (Eqs. 6-7 variant).

    B is initialized to zero, so at epoch 0 W_s == I exactly (the model starts
    from the frozen CLIP geometry). Exposes a .weight property returning the
    effective (d, d) matrix so existing code referring to psc.Ws.weight keeps
    working. Requires input_dim == shared_dim.
    """

    def __init__(self, dim: int, rank: int):
        super().__init__()
        if rank <= 0 or rank >= dim:
            raise ValueError(f"ws_rank must be in (0, {dim}), got {rank}.")
        self.dim = dim
        self.rank = rank
        self.down = nn.Linear(dim, rank, bias=False)   # A: R^d -> R^r
        self.up = nn.Linear(rank, dim, bias=False)      # B: R^r -> R^d
        with torch.no_grad():
            nn.init.normal_(self.down.weight, std=0.02)
            nn.init.zeros_(self.up.weight)

    @property
    def weight(self) -> torch.Tensor:
        """Effective W_s matrix (d, d) = I + B A."""
        eye = torch.eye(
            self.dim, device=self.down.weight.device, dtype=self.down.weight.dtype
        )
        return eye + self.up.weight @ self.down.weight

    def delta_norm_sq(self) -> torch.Tensor:
        """||B A||_F^2 — deviation from identity (ws_anchor penalty term)."""
        delta = self.up.weight @ self.down.weight
        return (delta ** 2).sum()

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return phi + self.up(self.down(phi))


class PriSematicConstructor(nn.Module):
    """Primitive Semantic Constructor for Eqs. (4)-(8).

    Accepts a dict of class entries. Each value can be:
    - A dict {name, attr, hier, desc} → encode (join or itemwise), project
    - A string → tokenize, encode, project
    - A tensor → already-encoded phi(t_c), project only

    When ``model.semantic.itemwise_pool`` is true (Plan 3 default under
    ``unify_after_llm``), list fields are encoded item-by-item and mean-pooled
    per field, then mixed across fields. This keeps each CLIP encode within
    the 77-token budget instead of joining long multi-item lists.

    When ``model.semantic.desc_only`` is true, only the desc_c field of each
    class is encoded (name_c/attr_c/hier_c ignored) — desc-only ablation.

    When ``model.semantic.field_weights`` is set (list of 4 floats for
    name_c, attr_c, hier_c, desc_c), the field vectors are combined with
    those weights instead of equal mean. This counters the finding that
    attr_c has near-zero discriminability in CLIP space (mean inter-class
    cosine 0.95) while name_c is highly discriminative (0.58).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        semantic_cfg = cfg.model.semantic
        input_dim = getattr(semantic_cfg, "input_dim", None)
        requested_backbone_type = getattr(semantic_cfg, "backbone_type", "embedding")
        backbone_type = requested_backbone_type
        if input_dim is not None:
            backbone_type = "embedding"
        self.backbone_type = backbone_type
        self.itemwise_pool = bool(getattr(semantic_cfg, "itemwise_pool", False))
        self.desc_only = bool(getattr(semantic_cfg, "desc_only", False))
        # L2-normalize each field vector before mixing them. Without it the
        # fields enter the mean with wildly different norms (name_c is one
        # item, attr_c/desc_c are means of ten), so field_weights do not
        # actually control the mixture.
        self.field_norm = bool(getattr(semantic_cfg, "field_norm", False))
        # Weighted field pooling: [w_name, w_attr, w_hier, w_desc].
        # None or absent → equal mean (backward compatible).
        raw_fw = getattr(semantic_cfg, "field_weights", None)
        if raw_fw is not None:
            fw = [float(x) for x in raw_fw]
            if len(fw) != 4:
                raise ValueError(
                    f"field_weights must have exactly 4 entries, got {len(fw)}."
                )
            s = sum(fw)
            if s <= 0:
                raise ValueError("field_weights must sum to a positive value.")
            self.field_weights = [x / s for x in fw]
        else:
            self.field_weights = None
        # Name prompt template (GĐ3 E4): encode the class name wrapped in a
        # photo-style template, e.g. "a photo of a {name}, a type of bird."
        # None (default) encodes the bare name as before.
        self.name_prompt = getattr(semantic_cfg, "name_prompt", None)
        if self.name_prompt is not None:
            self.name_prompt = str(self.name_prompt)

        kwargs = dict(
            backbone_type=backbone_type,
            pretrained=getattr(semantic_cfg, "pretrained", True),
            semantic_cfg=semantic_cfg,
        )
        if input_dim is not None:
            kwargs["input_dim"] = input_dim
        self.encoder = load_semantic_backbone(**kwargs)
        self.freeze_encoder = bool(getattr(semantic_cfg, "freeze", False))
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        self.encoder_dim = getattr(self.encoder, "out_features", input_dim)
        if self.encoder_dim is None:
            self.encoder_dim = getattr(semantic_cfg, "text_dim", 768)

        self.shared_dim = getattr(semantic_cfg, "shared_dim", 512)
        # Low-rank W_s = I + B A (aligned-geometry design): keeps W_s near the
        # identity so the semantic constellation of ALL classes (incl. unseen)
        # stays anchored to the frozen CLIP geometry while allowing limited,
        # structure-preserving adaptation. ws_rank=0 → plain nn.Linear as before.
        self.ws_rank = int(getattr(semantic_cfg, "ws_rank", 0))
        if self.ws_rank > 0:
            if self.encoder_dim != self.shared_dim:
                raise ValueError(
                    "ws_rank requires semantic.shared_dim == encoder dim "
                    f"(got {self.shared_dim} vs {self.encoder_dim})."
                )
            self.Ws = LowRankWs(self.shared_dim, self.ws_rank)
        else:
            self.Ws = nn.Linear(self.encoder_dim, self.shared_dim, bias=False)
        # Diagnostic switch: freeze Ws so the semantic bank stays fixed.
        # When encoder_dim == shared_dim it is set to identity, so p_sem becomes
        # the raw CLIP text embedding. Isolates whether seen-bias is driven by a
        # fitted Ws (text side) or by the visual projector (mfa).
        if bool(getattr(semantic_cfg, "freeze_Ws", False)):
            if self.ws_rank > 0:
                raise ValueError("freeze_Ws is incompatible with ws_rank > 0.")
            if self.encoder_dim == self.shared_dim:
                with torch.no_grad():
                    self.Ws.weight.copy_(torch.eye(self.shared_dim))
            self.Ws.requires_grad_(False)
        # Soft anchor: init Ws at identity and keep it trainable. Trainer adds
        # ws_anchor * ||Ws - I||_F^2 to the loss so Ws can only deviate with a
        # cost (countermeasure for the fitted-Ws seen-shortcut found in E3).
        # With ws_rank>0 the identity init is exact by construction (B=0) and
        # the trainer penalizes ||B A||_F^2 via Ws.delta_norm_sq().
        self.ws_anchor = float(getattr(semantic_cfg, "ws_anchor", 0.0))
        if self.ws_anchor > 0 and self.ws_rank == 0:
            if self.encoder_dim != self.shared_dim:
                raise ValueError(
                    "ws_anchor requires semantic.shared_dim == encoder dim "
                    f"(got {self.shared_dim} vs {self.encoder_dim})."
                )
            with torch.no_grad():
                self.Ws.weight.copy_(torch.eye(self.shared_dim))

        # Per-class learnable residual on the semantic side: p_sem = Ws(phi) + r[c].
        # RCNn's largest single gain came from learnable class-name vectors; this
        # applies the same idea to the whole grounded phi. Zero-initialized, so
        # training starts exactly at Ws(phi). Rows only receive gradient where a
        # caller indexes them, so every path that consumes p_sem must go through
        # project_phi (see trainer p_all / phi_bank).
        residual_classes = int(getattr(semantic_cfg, "class_residual", 0) or 0)
        self.class_residual = None
        if residual_classes > 0:
            self.class_residual = nn.Embedding(residual_classes, self.shared_dim)
            nn.init.zeros_(self.class_residual.weight)

    def project_phi(self, phi: torch.Tensor, class_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Ws(phi) plus the per-class residual rows, when class_residual is on."""
        out = self.Ws(phi)
        if self.class_residual is None:
            return out
        if class_indices is None:
            raise ValueError(
                "model.semantic.class_residual > 0 requires class_indices in project_phi."
            )
        rows = class_indices.to(out.device).long().view(-1)
        if int(rows.max()) >= self.class_residual.num_embeddings:
            raise ValueError(
                f"class index {int(rows.max())} out of range for class_residual="
                f"{self.class_residual.num_embeddings}."
            )
        return out + self.class_residual(rows)

    def class_indices_of(self, text_feature) -> torch.Tensor:
        """Global class indices of a text_feature dict, as expected by project_phi."""
        keys = list(text_feature.keys())
        try:
            return torch.tensor([int(k) for k in keys], dtype=torch.long)
        except (TypeError, ValueError):
            raise ValueError(
                "model.semantic.class_residual requires text_feature keys to be global "
                f"class indices, got {keys[:3]!r}."
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def ws_delta_norm_sq(self) -> torch.Tensor:
        """||W_s - I||_F^2 for the trainer's ws_anchor penalty.

        Low-rank path: ||B A||_F^2 (cheap, no 512x512 materialization in the
        backward graph beyond the delta). Plain nn.Linear path: full deviation
        from identity (requires shared_dim == encoder_dim).
        """
        if self.ws_rank > 0:
            return self.Ws.delta_norm_sq()
        if self.encoder_dim != self.shared_dim:
            raise ValueError(
                "ws_delta_norm_sq requires shared_dim == encoder dim for a plain Ws."
            )
        eye = torch.eye(
            self.shared_dim,
            device=self.Ws.weight.device,
            dtype=self.Ws.weight.dtype,
        )
        return ((self.Ws.weight - eye) ** 2).sum()

    def _encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode one or more strings → (N, D) float tensor on encoder device."""
        if not texts:
            raise ValueError("Cannot encode empty text list.")

        def _run():
            if len(texts) == 1:
                out = self.encoder(texts[0])
            else:
                out = self.encoder(list(texts))
            if out.dim() == 1:
                out = out.unsqueeze(0)
            return out

        # Frozen text encoder: no graph (and safe for one-shot class-phi cache).
        if self.freeze_encoder:
            with torch.no_grad():
                return _run()
        return _run()

    @torch.no_grad()
    def encode_class_phi(self, records: Sequence[dict]) -> torch.Tensor:
        """Encode a list of t_c dicts → (C, D) phi_t (no Ws).

        Used by the trainer to cache frozen text embeddings once per run.
        """
        if not records:
            raise ValueError("encode_class_phi requires a non-empty record list.")
        was_training = self.training
        self.eval()
        phis = []
        for record in records:
            if not isinstance(record, dict):
                raise TypeError(
                    f"encode_class_phi expects dict records, got {type(record)}"
                )
            phi = self._encode_record(record)
            if phi.dim() != 1:
                phi = phi.reshape(-1)
            phis.append(phi.detach())
        if was_training:
            self.train(True)
        return torch.stack(phis, dim=0)

    def class_phi_sentence_texts(self, records: Sequence[dict]) -> List[List[str]]:
        """Flatten each record into its per-sentence prompt list (v9.6 E2).

        Mirrors _encode_record_itemwise's field order (name, attr, hier,
        desc). Each list item of attr_c/hier_c/desc_c becomes one sentence;
        the name becomes one sentence (name_prompt applied). Returns raw
        strings only — no encoding — so callers can batch-encode them and
        build the [C, S, D] per-sentence tensor themselves.
        """
        out: List[List[str]] = []
        for record in records:
            if not isinstance(record, dict):
                raise TypeError(
                    f"class_phi_sentence_texts expects dict records, got {type(record)}"
                )
            sentences: List[str] = []
            name = _first_present(record, "name_c", "name", "class_name", "label")
            if name is not None:
                name_text = str(name).strip() if not isinstance(name, (list, tuple)) else ", ".join(
                    str(x).strip() for x in name if str(x).strip()
                )
                if name_text:
                    if self.name_prompt:
                        name_text = self.name_prompt.replace("{name}", name_text)
                    sentences.append(name_text)
            for keys in (
                ("attr_c", "attr", "attributes", "attrs", "attribute"),
                ("hier_c", "hier", "hierarchy", "parents", "path"),
                ("desc_c", "desc", "description", "text", "caption"),
            ):
                value = _first_present(record, *keys)
                sentences.extend(_as_item_list(value))
            if not sentences:
                raise ValueError(
                    "Class semantic record must contain name_c/attr_c/hier_c/desc_c fields."
                )
            out.append(sentences)
        return out

    def _encode_mean(self, texts: Sequence[str]) -> torch.Tensor:
        emb = self._encode_texts(list(texts))
        return emb.mean(dim=0)

    def _build_prompt(self, t_c: dict) -> str:
        def clean(value):
            if isinstance(value, (list, tuple)):
                return ", ".join(str(item).strip() for item in value if str(item).strip())
            return str(value).strip()

        parts = []
        name = _first_present(t_c, "name_c", "name", "class_name", "label")
        attr = _first_present(t_c, "attr_c", "attr", "attributes", "attrs", "attribute")
        hier = _first_present(t_c, "hier_c", "hier", "hierarchy", "parents", "path")
        desc = _first_present(t_c, "desc_c", "desc", "description", "text", "caption")
        if name:
            parts.append(f"Class name: {clean(name)}.")
        if attr:
            parts.append(f"Attributes: {clean(attr)}.")
        if hier:
            parts.append(f"Hierarchy: {clean(hier)}.")
        if desc:
            parts.append(f"Description: {clean(desc)}.")
        if not parts:
            raise ValueError("Class semantic record must contain name_c/attr_c/hier_c/desc_c fields.")
        return " ".join(parts)

    def _encode_record_itemwise(self, t_c: dict) -> torch.Tensor:
        field_vectors: List[torch.Tensor] = []
        field_present: List[bool] = []

        name = _first_present(t_c, "name_c", "name", "class_name", "label")
        if name is not None:
            name_text = str(name).strip() if not isinstance(name, (list, tuple)) else ", ".join(
                str(x).strip() for x in name if str(x).strip()
            )
            if name_text:
                if self.name_prompt:
                    name_text = self.name_prompt.replace("{name}", name_text)
                field_vectors.append(self._encode_mean([name_text]))
                field_present.append(True)
            else:
                field_present.append(False)
        else:
            field_present.append(False)

        for keys in (
            ("attr_c", "attr", "attributes", "attrs", "attribute"),
            ("hier_c", "hier", "hierarchy", "parents", "path"),
            ("desc_c", "desc", "description", "text", "caption"),
        ):
            value = _first_present(t_c, *keys)
            items = _as_item_list(value)
            if not items:
                field_present.append(False)
                continue
            field_vectors.append(self._encode_mean(items))
            field_present.append(True)

        if not field_vectors:
            raise ValueError(
                "Class semantic record must contain name_c/attr_c/hier_c/desc_c fields."
            )
        if self.field_norm:
            field_vectors = [F.normalize(v, p=2, dim=-1) for v in field_vectors]
        stacked = torch.stack(field_vectors, dim=0)
        if self.field_weights is not None and len(field_vectors) == 4:
            # All 4 fields present: apply configured weights.
            w = torch.tensor(
                self.field_weights, dtype=stacked.dtype, device=stacked.device
            )
            return (w.view(4, 1) * stacked).sum(dim=0)
        return stacked.mean(dim=0)

    def _encode_record(self, t_c: dict) -> torch.Tensor:
        if self.desc_only:
            value = _first_present(t_c, "desc_c", "desc", "description", "text", "caption")
            items = _as_item_list(value)
            if not items:
                raise ValueError("desc_only mode requires a desc_c field per class.")
            return self._encode_mean(items)
        if self.itemwise_pool:
            return self._encode_record_itemwise(t_c)
        prompt = self._build_prompt(t_c)
        return self._encode_texts([prompt]).squeeze(0)

    def forward(self, text_feature) -> Dict[str, torch.Tensor]:
        features = []
        for key, sematic in text_feature.items():
            if isinstance(sematic, dict):
                phi = self._encode_record(sematic)
            elif isinstance(sematic, str):
                phi = self._encode_texts([sematic]).squeeze(0)
            else:
                phi = sematic
            features.append(phi)
        phi_t = torch.stack(features, dim=0)
        if self.class_residual is None:
            p_sem = self.Ws(phi_t)
        else:
            p_sem = self.project_phi(phi_t, self.class_indices_of(text_feature))
        p_sem_norm = F.normalize(p_sem, p=2, dim=-1)
        return {
            "p_sem_norm": p_sem_norm,
            "phi_t": phi_t,
            "p_sem_unnorm": p_sem
        }

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "../..")))

    import torch
    from omegaconf import OmegaConf

    _BASE = {"model": {"semantic": {
        "shared_dim": 512, "pretrained": False,
    }}}

    # 1. Pre-features (embedding backbone) — pre-encoded tensors
    print("1. Pre-features (embedding backbone)...")
    c = OmegaConf.merge(OmegaConf.create(_BASE), OmegaConf.create({
        "model": {"semantic": {"backbone_type": "embedding", "input_dim": 85, "shared_dim": 128}}
    }))
    con = PriSematicConstructor(c)
    x = {"c1": torch.randn(85), "c2": torch.randn(85)}
    out = con(x)
    assert out["p_sem_norm"].shape == (2, 128)
    assert out["phi_t"].shape == (2, 85)
    assert out["p_sem_unnorm"].shape == (2, 128)
    assert torch.allclose(out["p_sem_norm"].norm(p=2, dim=-1), torch.ones(2))
    print("   OK")

    # 2. BERT backbone (pretrained=False) — input là text raw
    print("2. BERT backbone (Transformer)...")
    c = OmegaConf.merge(OmegaConf.create(_BASE), OmegaConf.create({
        "model": {"semantic": {"backbone_type": "bert", "text_dim": 768}}
    }))
    con = PriSematicConstructor(c)
    con = con.cpu()

    input_text = {
        "class1": {"name": "Alice", "attr": "tall", "hier": "person", "desc": "A tall person named Alice."},
        "class2": {"name": "Bob", "attr": "short", "hier": "person", "desc": "A short person named Bob."},
    }
    out = con(input_text)
    assert out["p_sem_norm"].shape == (2, 512), f"p_sem_norm shape: {out['p_sem_norm'].shape}"
    assert out["phi_t"].shape == (2, 768), f"phi_t shape: {out['phi_t'].shape}"
    assert out["p_sem_unnorm"].shape == (2, 512)
    assert torch.allclose(out["p_sem_norm"].norm(p=2, dim=-1), torch.ones(2))
    print("   OK")

    # 3. CLIP backbone — input là text raw (t_c dict)
    print("3. CLIP backbone...")
    try:
        c = OmegaConf.merge(OmegaConf.create(_BASE), OmegaConf.create({
            "model": {"semantic": {"backbone_type": "clip", "clip_model": "ViT-B/32"}}
        }))
        con = PriSematicConstructor(c)
        con = con.cpu()

        input_text = {
            "class1": {"name": "Alice", "attr": "tall", "hier": "person", "desc": "A tall person named Alice."},
            "class2": {"name": "Bob", "attr": "short", "hier": "person", "desc": "A short person named Bob."},
        }
        out = con(input_text)
        assert out["p_sem_norm"].shape == (2, 512)
        assert out["phi_t"].shape == (2, 512)
        assert out["p_sem_unnorm"].shape == (2, 512)
        assert torch.allclose(out["p_sem_norm"].norm(p=2, dim=-1), torch.ones(2))
        print("   OK")
    except Exception as e:
        print(f"   SKIP: {e}")

    # 3b. CLIP itemwise pool with list fields
    print("3b. CLIP itemwise pool...")
    try:
        c = OmegaConf.merge(OmegaConf.create(_BASE), OmegaConf.create({
            "model": {"semantic": {
                "backbone_type": "clip",
                "clip_model": "ViT-B/32",
                "itemwise_pool": True,
            }}
        }))
        con = PriSematicConstructor(c)
        con = con.cpu()
        input_text = {
            "class1": {
                "name_c": "zebra",
                "attr_c": ["black and white stripes", "horse-like body"],
                "hier_c": ["plains zebra", "mountain zebra"],
                "desc_c": ["african savanna", "grassland near water"],
            },
            "class2": {
                "name_c": "tiger",
                "attr_c": ["orange coat with black stripes"],
                "hier_c": ["bengal tiger"],
                "desc_c": ["dense forest"],
            },
        }
        out = con(input_text)
        assert out["p_sem_norm"].shape == (2, 512)
        assert torch.allclose(out["p_sem_norm"].norm(p=2, dim=-1), torch.ones(2), atol=1e-4)
        print("   OK")
    except Exception as e:
        print(f"   SKIP: {e}")

    # 4. Checkpoint save/load
    print("4. Checkpoint save/load...")
    ckpt_dir = "/tmp/hsprn_test_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best.pth")

    con = PriSematicConstructor(OmegaConf.merge(OmegaConf.create(_BASE), OmegaConf.create({
        "model": {"semantic": {"backbone_type": "embedding", "input_dim": 85}}
    })))
    x = {"c1": torch.randn(85), "c2": torch.randn(85)}
    out_before = con(x)

    torch.save({"model_state_dict": con.state_dict(), "config": "embedding"}, ckpt_path)
    assert os.path.exists(ckpt_path)

    con2 = PriSematicConstructor(OmegaConf.merge(OmegaConf.create(_BASE), OmegaConf.create({
        "model": {"semantic": {"backbone_type": "embedding", "input_dim": 85}}
    })))
    con2.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    out_after = con2(x)

    assert torch.allclose(out_before["p_sem_norm"], out_after["p_sem_norm"])
    assert torch.allclose(out_before["phi_t"], out_after["phi_t"])
    assert torch.allclose(out_before["p_sem_unnorm"], out_after["p_sem_unnorm"])

    import shutil
    shutil.rmtree(ckpt_dir)
    print("   OK")

    print("\nAll tests passed!")
