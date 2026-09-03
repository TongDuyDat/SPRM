"""T9b — verify build_prepared_zsl_splits honors semantic_embedding_path.

Monkeypatches build_prepared_zsl_dataset (no real CUB package needed) and checks:
  1. semantic_embedding_path=None  -> attribute = package attribute (backward compat).
  2. semantic_embedding_path set   -> attribute overridden by the .pt embeddings.
  3. class-count mismatch          -> ValueError.
"""
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np
import torch

import datasets.prepared_dataset as pd_mod
from datasets.prepared_dataset import build_prepared_zsl_splits

NUM_CLASSES = 10
PKG_DIM = 312      # package (xlsa17) attribute dim
PHI_DIM = 512      # precomputed phi dim

results = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f"  -> {detail}" if detail else ""))
    return cond


class _MockDS:
    def __init__(self):
        self.attribute = np.random.randn(NUM_CLASSES, PKG_DIM).astype(np.float32)
        self.seen_classes = np.arange(6)
        self.unseen_classes = np.arange(6, NUM_CLASSES)

    @property
    def num_classes(self):
        return self.attribute.shape[0]


# Patch the dataset builder so no real package is read.
pd_mod.build_prepared_zsl_dataset = lambda *a, **k: _MockDS()

# --- 1. Backward compat: no semantic_embedding_path -----------------------
splits = build_prepared_zsl_splits("dummy_root", "CUB")
check("no path -> attribute is package attribute (312-dim)",
      splits["attribute"].shape == (NUM_CLASSES, PKG_DIM),
      f"shape={tuple(splits['attribute'].shape)}")
check("no path -> semantic_embedding == attribute",
      torch.equal(splits["semantic_embedding"], splits["attribute"]))

# --- 2. Override: semantic_embedding_path set -----------------------------
phi = torch.randn(NUM_CLASSES, PHI_DIM)
with tempfile.TemporaryDirectory() as td:
    pt_path = os.path.join(td, "phi.pt")
    torch.save({"embeddings": phi, "class_semantics": [{"name_c": f"c{i}"} for i in range(NUM_CLASSES)]}, pt_path)

    splits = build_prepared_zsl_splits("dummy_root", "CUB", semantic_embedding_path=pt_path)
    check("path set -> attribute overridden by phi (512-dim)",
          splits["attribute"].shape == (NUM_CLASSES, PHI_DIM),
          f"shape={tuple(splits['attribute'].shape)}")
    check("path set -> values match saved phi",
          torch.allclose(splits["attribute"], phi.float()))
    check("path set -> semantic_embedding == attribute",
          torch.equal(splits["semantic_embedding"], splits["attribute"]))
    check("path set -> splits/datasets still returned",
          splits["trainval"] is not None and splits["test_seen"] is not None
          and splits["test_unseen"] is not None)

    # --- 3. Class-count mismatch -> ValueError ----------------------------
    bad = torch.randn(NUM_CLASSES - 1, PHI_DIM)
    bad_path = os.path.join(td, "bad.pt")
    torch.save({"embeddings": bad}, bad_path)
    try:
        build_prepared_zsl_splits("dummy_root", "CUB", semantic_embedding_path=bad_path)
        check("class-count mismatch raises ValueError", False, "no exception raised")
    except ValueError as e:
        check("class-count mismatch raises ValueError", True, str(e)[:70])

print("\n" + "=" * 60)
n_pass = sum(1 for r in results if r[0] == "PASS")
print(f"T9b RESULT: {n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    for s, n, d in results:
        if s == "FAIL":
            print(f"  - {n}: {d}")
    sys.exit(1)
print("ALL PASS")
