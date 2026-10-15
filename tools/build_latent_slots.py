#!/usr/bin/env python
"""Sinh slot latent cho LPR: Z0 (k-means) và bảng ái lực A .

Chỉ dùng lớp SEEN (trainval ∪ test_seen). Lớp unseen bị loại ngay từ đầu và có
assert, vì k-means trên câu của lớp unseen làm setting không còn là inductive ZSL.

Đầu ra (mặc định cạnh nhau trong pre-trained/):
    <ds>_slots_k<K>_<tag>.pt    {"embeddings": Z0 [K, d], "meta": {...}}   -> refiner.slot_path
    <ds>_slottgt_k<K>_<tag>.pt  {"A": [C, K], "seen_classes": [...], ...}  -> refiner.slot_target_path
Z0 cùng định dạng với *_attrq_b16.pt nên refiner đọc được như attribute bank.

    python tools/build_latent_slots.py --dataset cub --n_slots 64 --device cuda

Phần lõi (chọn lớp seen, chọn câu, k-means, ái lực) thuần torch/numpy, test được
không cần CLIP: xem tests/test_build_latent_slots.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

#: Trần số câu mỗi lớp. AWA2/SUN/aPY có 462–1208 câu attr_c mỗi lớp; encode toàn
#: bộ vài chục nghìn câu không đổi chất lượng cụm nhưng tốn thời gian.
MAX_SENTENCES_PER_CLASS = 64


def seen_classes_from_images(images_jsonl):
    """Lớp seen = nhãn của trainval ∪ test_seen. Trả (seen, unseen) đã sắp xếp."""
    seen, unseen = set(), set()
    with open(images_jsonl, encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            (unseen if rec["split"] == "test_unseen" else seen).add(int(rec["label"]))
    overlap = seen & unseen
    if overlap:
        raise AssertionError(f"lớp vừa seen vừa unseen: {sorted(overlap)[:5]}")
    return sorted(seen), sorted(unseen)


def class_sentences(record, max_n=MAX_SENTENCES_PER_CLASS, fields=("attr_c",)):
    """Câu mô tả thị giác của một lớp, lấy đều (không lấy n câu đầu) nếu quá dài."""
    out = []
    for f in fields:
        v = record.get(f)
        if isinstance(v, str):
            v = [s for s in v.splitlines() if s.strip()]
        out += [str(s).strip() for s in (v or []) if str(s).strip()]
    if len(out) > max_n:
        idx = np.linspace(0, len(out) - 1, max_n).round().astype(int)
        out = [out[i] for i in sorted(set(idx.tolist()))]
    return out


def collect_sentences(class_semantics, seen_classes, unseen_classes, max_n, fields):
    """(texts, owner) với owner[i] = chỉ số lớp của câu i. Chỉ lớp seen."""
    texts, owner = [], []
    for c in seen_classes:
        for s in class_sentences(class_semantics[c], max_n, fields):
            texts.append(s)
            owner.append(c)
    assert not (set(owner) & set(unseen_classes)), "câu của lớp unseen lọt vào bộ sinh slot"
    assert set(owner) <= set(seen_classes)
    return texts, owner


def spherical_kmeans(X, k, seed=42, n_init=10, iters=100):
    """Spherical k-means trên X [n, d] đã chuẩn hoá. Trả (centers [k, d], inertia)."""
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    if n < k:
        raise ValueError(f"{n} câu < K={k}")
    best_c, best_obj = None, -1.0
    for _ in range(n_init):
        # k-means++ trên khoảng cách cosine
        centers = [X[torch.randint(n, (1,), generator=g)].squeeze(0)]
        for _ in range(k - 1):
            d2 = (1 - torch.stack(centers) @ X.t()).min(0).values.clamp_min(0) ** 2
            p = d2 / d2.sum() if d2.sum() > 0 else torch.full((n,), 1.0 / n)
            centers.append(X[torch.multinomial(p, 1, generator=g)].squeeze(0))
        C = torch.stack(centers)
        for _ in range(iters):
            assign = (X @ C.t()).argmax(1)
            newC = torch.zeros_like(C).index_add_(0, assign, X)
            empty = newC.norm(dim=1) < 1e-8
            if empty.any():                      # cụm rỗng -> điểm xa tâm nhất
                far = (X @ C.t()).max(1).values.argsort()[: int(empty.sum())]
                newC[empty] = X[far]
            newC = F.normalize(newC, dim=-1)
            if torch.allclose(newC, C, atol=1e-6):
                C = newC
                break
            C = newC
        obj = float((X @ C.t()).max(1).values.sum())   # tổng cosine tới tâm gần nhất
        if obj > best_obj:
            best_c, best_obj = C, obj
    return best_c, best_obj


def affinity(sent_emb, owner, centers, n_classes, seen_classes):
    """A[c, k] = max_{câu e của lớp c} cos(e, z_k). Hàng của lớp unseen = 0."""
    A = torch.zeros(n_classes, centers.shape[0])
    owner = torch.as_tensor(owner)
    sims = sent_emb @ centers.t()
    for c in seen_classes:
        A[c] = sims[owner == c].max(0).values
    return A


def standardize_target(A, seen_classes):
    """L_attr target from the raw affinity: relu of the per-slot z-score.

    Raw A is near-constant across classes (every class has some sentence close
    to every slot): mean pairwise cosine between class rows is 0.98-0.998 on
    CUB/AWA2/SUN/aPY, so cos(attr_pred, A[y]) cannot tell classes apart. The
    per-slot z-score keeps only "this class is closer to slot k than classes
    usually are"; relu makes it non-negative like the human attribute rows.
    Pairwise cosine drops to 0.15-0.27 (human attributes: 0.27-0.60).
    Statistics come from seen rows only; unseen rows stay 0.
    """
    S = A[seen_classes]
    z = (S - S.mean(0, keepdim=True)) / S.std(0, keepdim=True).clamp_min(1e-6)
    T = torch.zeros_like(A)
    T[seen_classes] = torch.relu(z)
    return T


def mean_pair_cos(X):
    """Mean cosine between distinct rows (1.0 = all rows identical)."""
    Xn = F.normalize(X, dim=-1)
    n = X.shape[0]
    return float(((Xn @ Xn.t()).sum() - n) / (n * n - n))


def row_entropy(A, seen_classes):
    """Entropy chuẩn hoá [0,1] của softmax hàng A. Gần 1 = hàng phẳng."""
    P = A[seen_classes].softmax(-1)
    return float((-(P * P.clamp_min(1e-12).log()).sum(-1) / np.log(P.shape[-1])).mean())


# ---------------------------------------------------------------------------
# Per-family sources: (class_semantics rows in eval order, seen ids, unseen ids)
# ---------------------------------------------------------------------------

def source_classic(cfg, prepared_root=None, semantics_path=None):
    """Rows follow prepared_datasets/<DS>/metadata/classes.json. ``semantics_path``
    (e.g. data_semantics_v2/CUB/class_descriptions_cub_wordnet_after_llm.json,
    keyed by class name) replaces the package's class_semantics.json."""
    root = Path(prepared_root or getattr(cfg.data, "prepared_root", "./prepared_datasets"))
    meta = root / cfg.data.dataset / "metadata"
    seen, unseen = seen_classes_from_images(meta / "images.jsonl")
    if semantics_path:
        classes = sorted(json.loads((meta / "classes.json").read_text(encoding="utf-8")),
                         key=lambda c: int(c["class_index"]))
        sem = _records_in_order(semantics_path, [str(c["class_name"]) for c in classes])
    else:
        sem = json.loads((meta / "class_semantics.json").read_text(encoding="utf-8"))
    return sem, seen, unseen


def source_coco(cfg, semantics_path=None):
    """Rows follow bench6_classes/coco_tags.txt == the 65 eval columns in id order."""
    import pickle
    root = Path(cfg.data.data_root)
    split = pickle.load(open(root / "anno" / "cls_ids.pickle", "rb"))
    eval_ids = sorted(list(split["train"]) + list(split["test"]))
    pos = {c: j for j, c in enumerate(eval_ids)}
    seen = sorted(pos[c] for c in split["train"])
    unseen = sorted(pos[c] for c in split["test"])
    sem = json.loads(Path(semantics_path or "data_semantics/BENCH6/coco_class_semantics.json")
                     .read_text(encoding="utf-8"))
    if isinstance(sem, dict):   # data_semantics_v2: {tag: record}; order = coco_tags.txt
        tags = [t.strip() for t in (_ROOT / "bench6_classes" / "coco_tags.txt")
                .read_text(encoding="utf-8").splitlines() if t.strip()]
        sem = _records_in_order(semantics_path or "data_semantics_v2/BENCH6/coco_class_semantics.json", tags)
    if len(sem) != len(eval_ids):
        raise ValueError(f"COCO semantics has {len(sem)} rows, eval space has {len(eval_ids)}.")
    return sem, seen, unseen


def source_czsl(cfg, semantics_path=None):
    """Seen = pairs of the TRAIN split only (val/test pairs are never seen).

    Rows follow the closed pair universe of build_czsl_bench (sorted pairs),
    which is the order of bench6_classes/<ds>_pairs.txt and of the phi file.
    """
    from datasets.czsl_dataset import build_czsl_bench
    ds = str(cfg.data.dataset)
    bench = build_czsl_bench(ds, str(cfg.data.data_root), universe="closed")
    seen = [int(c) for c in bench["seen_classes"]]
    unseen = [int(c) for c in bench["unseen_classes"]]
    sem = json.loads(Path(semantics_path or f"data_semantics/BENCH6/{ds}_class_semantics.json")
                     .read_text(encoding="utf-8"))
    if len(sem) != bench["num_pairs"]:
        raise ValueError(f"{ds} semantics has {len(sem)} rows, pair universe has "
                         f"{bench['num_pairs']}.")
    return sem, seen, unseen


def _records_in_order(path, names):
    """Records of ``names`` in that order, from a list or a {name: record} JSON.

    generate_descriptions.py writes a dict keyed by class name; the older
    regenerate script writes a list. A missing or failed (``_error``) record is
    an error, not an empty slot source.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {r["name_c"]: r for r in data}
    bad = [n for n in names if n not in data or "_error" in data[n]]
    if bad:
        raise ValueError(f"{path}: {len(bad)} class(es) missing or failed, e.g. {bad[:5]}")
    return [data[n] for n in names]

def source_nuswide(cfg, semantics_path=None):
    """Columns follow datasets.builders: 925 seen tags (TagList1k \\ Concepts81)
    then the 81 unseen concepts. Only the seen tags' sentences reach k-means;
    unseen rows are placeholders so row indices match the score columns."""
    from datasets.builders import _nus_tag_lists
    seen925, c81 = _nus_tag_lists(str(Path(cfg.data.data_root) / "Groundtruth"))
    sem = _records_in_order(
        semantics_path or "data_semantics_v2/BENCH6/nuswide-seen_class_semantics.json",
        seen925)
    sem += [{"name_c": n} for n in c81]
    return sem, list(range(925)), list(range(925, 1006))

def load_source(cfg, prepared_root=None, semantics_path=None):
    source = str(getattr(cfg.data, "source", "prepared")).lower()
    if source in ("prepared", "legacy"):
        return source_classic(cfg, prepared_root, semantics_path)
    if source == "ml":
        name = str(cfg.data.dataset).lower()
        if name == "nuswide":
            return source_nuswide(cfg, semantics_path)
        if name != "coco":
            raise ValueError(f"multi-label slots: unknown dataset {name!r}")
        return source_coco(cfg, semantics_path)
    if source == "czsl":
        return source_czsl(cfg, semantics_path)
    raise ValueError(f"unknown data.source {source!r}")


def _text_encoder_cfg(cfg):
    """Config for the frozen CLIP text tower (Ws is built but unused for encoding)."""
    from types import SimpleNamespace
    return SimpleNamespace(model=SimpleNamespace(semantic=SimpleNamespace(
        backbone_type="clip", pretrained=True, clip_model=cfg.model.vision.clip_model,
        max_text_len=77, input_dim=None, shared_dim=512, ws_rank=32)))


def encode_sentences(cfg, texts, device):
    from model.sematics.constructor import PriSematicConstructor

    psc = PriSematicConstructor(_text_encoder_cfg(cfg)).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            out.append(psc._encode_texts(texts[i:i + 256]).float().cpu())
    return F.normalize(torch.cat(out), dim=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="config/datasets/<dataset>.yaml (e.g. cub)")
    ap.add_argument("--method", default="lpr")
    ap.add_argument("--n_slots", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fields", default="attr_c", help="Trường lấy câu, ngăn cách bởi dấu phẩy")
    ap.add_argument("--max_sentences", type=int, default=MAX_SENTENCES_PER_CLASS)
    ap.add_argument("--slot_init", choices=("kmeans", "random"), default="kmeans",
                    help="kmeans = the method (Eq. 3). random = ablation: K random seen-class "
                         "sentences as slots, everything else unchanged.")
    ap.add_argument("--tag", default="b16")
    ap.add_argument("--output_dir", default="pre-trained")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prepared_root", default=None)
    ap.add_argument("--semantics", default=None,
                    help="Records JSON that replaces the default (classic: data_semantics_v2/<DS>/...json; "
                         "COCO/CZSL: data_semantics/BENCH6/<ds>_class_semantics.json)")
    args = ap.parse_args()

    from tools.utils import compose_config
    cfg = compose_config(args.dataset, args.method)[0]
    ds = cfg.data.dataset
    sem, seen, unseen = load_source(cfg, args.prepared_root, args.semantics)
    n_classes = len(sem)
    assert not set(seen) & set(unseen), "seen ∩ unseen"
    assert len(seen) + len(unseen) == n_classes, (len(seen), len(unseen), n_classes)

    texts, owner = collect_sentences(sem, seen, unseen, args.max_sentences,
                                     tuple(args.fields.split(",")))
    print(f"{ds}: {len(seen)} seen / {len(unseen)} unseen | {len(texts)} câu "
          f"({len(texts) / len(seen):.1f}/lớp seen) | K={args.n_slots}")
    emb = encode_sentences(cfg, texts, args.device)
    if args.slot_init == "random":
        g = torch.Generator().manual_seed(args.seed)
        pick = torch.randperm(emb.shape[0], generator=g)[: args.n_slots]
        Z0 = emb[pick].clone()
        obj = float((emb @ Z0.t()).max(1).values.sum())
    else:
        Z0, obj = spherical_kmeans(emb, args.n_slots, seed=args.seed)
    A_raw = affinity(emb, owner, Z0, n_classes, seen)
    A = standardize_target(A_raw, seen)
    sep_raw, sep = mean_pair_cos(A_raw[seen]), mean_pair_cos(A[seen])
    sizes = torch.bincount((emb @ Z0.t()).argmax(1), minlength=args.n_slots)
    print(f"  cosine TB tới tâm = {obj / len(texts):.4f} | cụm nhỏ nhất/lớn nhất = "
          f"{int(sizes.min())}/{int(sizes.max())}")
    print(f"  cos TB giữa 2 lớp: A thô = {sep_raw:.3f} -> mục tiêu = {sep:.3f}"
          + ("  ⚠ mục tiêu chưa phân biệt lớp (>0.6)" if sep > 0.6 else ""))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{ds.lower()}_%s_k{args.n_slots}_{args.tag}.pt"
    info = {"dataset": ds, "n_slots": args.n_slots, "seed": args.seed, "fields": args.fields,
            "seen_classes": seen, "n_sentences": len(texts),
            "source": "spherical k-means" if args.slot_init == "kmeans" else "random sentences (ablation)"}
    torch.save({"embeddings": Z0, "meta": info}, out / (stem % "slots"))
    torch.save({"A": A, "A_raw": A_raw, "seen_classes": seen, "target": "relu(zscore per slot)",
                "pair_cos_raw": sep_raw, "pair_cos": sep, **info}, out / (stem % "slottgt"))
    print(f"  -> {out / (stem % 'slots')}\n  -> {out / (stem % 'slottgt')}")


if __name__ == "__main__":
    main()
