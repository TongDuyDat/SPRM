#!/usr/bin/env python
"""v9.6 E4 diagnostic: does prototype-side density r(c) predict top-1 win counts?

Motivation: raw and centered CSLS both collapsed T1 (74.4 -> 46.8 / 29.0),
meaning the hub term overwhelms the class signal. Before tuning any beta we
must measure whether the hub signal exists AT ALL in this pipeline:
Spearman corr between r(c) (mean cos to k nearest other prototypes, after Ws)
and the actual top-1 win count of class c on the unseen test set.
"""
import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.local_pkg import import_local_datasets
from tools.utils import load_config, set_seed
from tools.semantic_policy import configure_semantic_input, resolve_semantic_source

_ds = import_local_datasets()
build_prepared_zsl_splits = _ds.build_prepared_zsl_splits

from model.hsprn_model import HSPRN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ks", default="5,10,20")
    p.add_argument("--group", choices=("unseen", "seen", "both"), default="both")
    args = p.parse_args()

    cfg = load_config(args.config, default_path=os.path.join(
        os.path.dirname(args.config), "default.yaml"))
    set_seed(getattr(cfg.training, "seed", 42))
    device = args.device

    splits = build_prepared_zsl_splits(
        cfg.data.prepared_root, cfg.data.dataset,
        require_full_semantics=True,
        semantic_embedding_path=cfg.data.semantic_embedding_path,
        image_size=getattr(cfg.data, "image_size", 224),
        image_preprocess=getattr(cfg.data, "image_preprocess", None) or "clip",
    )
    test_unseen = splits["test_unseen"]
    attribute = splits["attribute"]
    seen_classes = list(map(int, splits["seen_classes"]))
    unseen_classes = list(map(int, splits["unseen_classes"]))
    configure_semantic_input(cfg, attribute, "embedding")

    model = HSPRN(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    all_classes = seen_classes + unseen_classes
    phi_cache = attribute.detach().to(device)
    attr = {str(int(c)): phi_cache[int(c)] for c in all_classes}

    loader = DataLoader(test_unseen, batch_size=256, shuffle=False, num_workers=8)
    counts_u = {c: 0 for c in unseen_classes}
    loader_s = DataLoader(splits["test_seen"], batch_size=256, shuffle=False, num_workers=8)
    counts_s = {c: 0 for c in seen_classes}

    with torch.no_grad():
        for batch in loader:
            features, labels = batch[:2]
            out = model(attr, features.to(device))
            preds = out.scores_sem.argmax(dim=-1).cpu()
            for pr in preds.tolist():
                c = all_classes[pr]
                if c in counts_u:
                    counts_u[c] += 1
        for batch in loader_s:
            features, labels = batch[:2]
            out = model(attr, features.to(device))
            preds = out.scores_sem.argmax(dim=-1).cpu()
            for pr in preds.tolist():
                c = all_classes[pr]
                if c in counts_s:
                    counts_s[c] += 1

        base_u = phi_cache[torch.tensor(unseen_classes, device=device)]
        protos_u = F.normalize(model.psc.Ws(base_u), p=2, dim=-1)
        cos_u = protos_u @ protos_u.T
        off_u = cos_u.clone()
        off_u.fill_diagonal_(-1.0)
        base_s = phi_cache[torch.tensor(seen_classes, device=device)]
        protos_s = F.normalize(model.psc.Ws(base_s), p=2, dim=-1)
        cos_s = protos_s @ protos_s.T
        off_s = cos_s.clone()
        off_s.fill_diagonal_(-1.0)

    def spearman(a, b):
        ra = torch.tensor(a, dtype=torch.float64).argsort().argsort().double()
        rb = torch.tensor(b, dtype=torch.float64).argsort().argsort().double()
        ra = (ra - ra.mean()) / ra.std()
        rb = (rb - rb.mean()) / rb.std()
        return float((ra * rb).mean())

    def report(tag, counts, off, classes):
        win = [counts[c] for c in classes]
        n = len(win)
        print(f"[{tag}] win counts: mean={sum(win)/n:.1f} max={max(win)} "
              f"min={min(win)} (hub ratio {max(win)/(sum(win)/n):.2f})")
        for k in [int(x) for x in args.ks.split(",")]:
            r = off.topk(k, dim=-1).values.mean(dim=-1).cpu().tolist()
            rho = spearman(r, win)
            print(f"[{tag}] k={k:2d}: Spearman(r(c), win-count) = {rho:+.3f}")

    if args.group in ("unseen", "both"):
        report("unseen", counts_u, off_u, unseen_classes)
    if args.group in ("seen", "both"):
        report("seen", counts_s, off_s, seen_classes)


if __name__ == "__main__":
    main()
