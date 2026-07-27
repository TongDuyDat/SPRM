#!/usr/bin/env python
"""Pack prepared_datasets/<DATASET> into colab_upload/<ds>_prepared.tar.gz."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CUB"],
        help="Dataset package names under prepared_datasets/ (default: CUB)",
    )
    parser.add_argument(
        "--prepared-root",
        default="prepared_datasets",
        help="Root containing per-dataset packages",
    )
    parser.add_argument(
        "--output-dir",
        default="colab_upload",
        help="Where to write <ds>_prepared.tar.gz",
    )
    args = parser.parse_args()

    prepared_root = Path(args.prepared_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        pkg = prepared_root / name
        manifest = pkg / "manifest.yaml"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing prepared package: {pkg}")
        out = out_dir / f"{name.lower()}_prepared.tar.gz"
        print(f"Packing {pkg} → {out}")
        with tarfile.open(out, "w:gz", compresslevel=1) as tf:
            tf.add(pkg, arcname=name)
        size_gb = out.stat().st_size / (1024**3)
        print(f"  OK {out} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
