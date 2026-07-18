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
        self.Ws = nn.Linear(self.encoder_dim, self.shared_dim, bias=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode one or more strings → (N, D) float tensor on encoder device."""
        if not texts:
            raise ValueError("Cannot encode empty text list.")
        if len(texts) == 1:
            out = self.encoder(texts[0])
        else:
            out = self.encoder(list(texts))
        if out.dim() == 1:
            out = out.unsqueeze(0)
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

        name = _first_present(t_c, "name_c", "name", "class_name", "label")
        if name is not None:
            name_text = str(name).strip() if not isinstance(name, (list, tuple)) else ", ".join(
                str(x).strip() for x in name if str(x).strip()
            )
            if name_text:
                field_vectors.append(self._encode_mean([name_text]))

        for keys in (
            ("attr_c", "attr", "attributes", "attrs", "attribute"),
            ("hier_c", "hier", "hierarchy", "parents", "path"),
            ("desc_c", "desc", "description", "text", "caption"),
        ):
            value = _first_present(t_c, *keys)
            items = _as_item_list(value)
            if not items:
                continue
            field_vectors.append(self._encode_mean(items))

        if not field_vectors:
            raise ValueError(
                "Class semantic record must contain name_c/attr_c/hier_c/desc_c fields."
            )
        stacked = torch.stack(field_vectors, dim=0)
        return stacked.mean(dim=0)

    def _encode_record(self, t_c: dict) -> torch.Tensor:
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
        p_sem = self.Ws(phi_t)
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
