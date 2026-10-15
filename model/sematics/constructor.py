from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import load_semantic_backbone


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
    """Primitive Semantic Constructor (Eqs. 4-8).

    ``forward`` takes a dict of class entries; each value is
    - a dict {name_c, attr_c, hier_c, desc_c} -> encoded by the frozen text backbone,
    - a str                                   -> encoded as one sentence,
    - a tensor                                -> already-encoded phi(t_c),
    then projected by the low-rank W_s = I + B A. The semantic encoder is always frozen.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        sc = cfg.model.semantic
        input_dim = getattr(sc, "input_dim", None)
        backbone_type = "embedding" if input_dim is not None else getattr(sc, "backbone_type", "embedding")
        self.backbone_type = backbone_type

        kwargs = dict(
            backbone_type=backbone_type,
            pretrained=getattr(sc, "pretrained", True),
            semantic_cfg=sc,
        )
        if input_dim is not None:
            kwargs["input_dim"] = input_dim
        self.encoder = load_semantic_backbone(**kwargs)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        self.encoder_dim = getattr(self.encoder, "out_features", input_dim)
        self.shared_dim = getattr(sc, "shared_dim", 512)
        self.ws_rank = int(sc.ws_rank)
        if self.ws_rank <= 0 or self.encoder_dim != self.shared_dim:
            raise ValueError(
                "low-rank Ws needs model.semantic.ws_rank > 0 and shared_dim == encoder dim "
                f"(got ws_rank={self.ws_rank}, shared_dim={self.shared_dim}, "
                f"encoder_dim={self.encoder_dim})."
            )
        self.Ws = LowRankWs(self.shared_dim, self.ws_rank)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def ws_delta_norm_sq(self) -> torch.Tensor:
        """||B A||_F^2: deviation of W_s from identity (trainer's ws_anchor penalty)."""
        return self.Ws.delta_norm_sq()

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

        # Frozen text encoder: no graph.
        with torch.no_grad():
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

    def _encode_mean(self, texts: Sequence[str]) -> torch.Tensor:
        emb = self._encode_texts(list(texts))
        return emb.mean(dim=0)

    def _encode_record(self, t_c: dict) -> torch.Tensor:
        """phi(t_c): field-mean of raw sentence embeddings (not Eq. 1 sentence-mean).

        Each field (name, attr, hier, desc) is the mean of its un-normalised sentence
        embeddings; the record is the mean of the present field vectors.
        """
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
            items = _as_item_list(_first_present(t_c, *keys))
            if items:
                field_vectors.append(self._encode_mean(items))

        if not field_vectors:
            raise ValueError(
                "Class semantic record must contain name_c/attr_c/hier_c/desc_c fields."
            )
        return torch.stack(field_vectors, dim=0).mean(dim=0)

    def forward(self, text_feature) -> Dict[str, torch.Tensor]:
        features = []
        for _, sematic in text_feature.items():
            if isinstance(sematic, dict):
                phi = self._encode_record(sematic)
            elif isinstance(sematic, str):
                phi = self._encode_texts([sematic]).squeeze(0)
            else:
                phi = sematic
            features.append(phi)
        phi_t = torch.stack(features, dim=0)
        p_sem = self.Ws(phi_t)
        return {
            "p_sem_norm": F.normalize(p_sem, p=2, dim=-1),
            "phi_t": phi_t,
            "p_sem_unnorm": p_sem,
        }
