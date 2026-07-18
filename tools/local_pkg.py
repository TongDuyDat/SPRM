"""Force-import project packages that collide with site-packages names.

Colab / pip may install HuggingFace ``datasets``. This repo also has a local
package named ``datasets``, so a normal import can hit the wrong module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parent.parent


def import_local_datasets() -> ModuleType:
    """Load repo-local ``datasets`` (never HuggingFace).

    Registers the module as ``sys.modules['datasets']`` and returns it.
    """
    pkg_dir = _ROOT / "datasets"
    init_py = pkg_dir / "__init__.py"
    prepared_py = pkg_dir / "prepared_dataset.py"
    zsl_py = pkg_dir / "zsl_dataset.py"

    missing = [p.name for p in (init_py, prepared_py, zsl_py) if not p.is_file()]
    if missing:
        raise ImportError(
            "Local datasets package incomplete under "
            f"{pkg_dir}. Missing: {missing}. "
            "Ensure datasets/*.py are present in the clone."
        )

    # Evict any previously imported package named datasets.
    for key in list(sys.modules):
        if key == "datasets" or key.startswith("datasets."):
            del sys.modules[key]

    root_s = str(_ROOT)
    sys.path = [p for p in sys.path if p not in (root_s, "")]
    sys.path.insert(0, root_s)

    spec = importlib.util.spec_from_file_location(
        "datasets",
        str(init_py),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {init_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["datasets"] = module
    spec.loader.exec_module(module)

    f = getattr(module, "__file__", "") or ""
    if "site-packages" in f or "dist-packages" in f:
        raise ImportError(
            f"Loaded wrong datasets from {f}. "
            "Remove leftover: rm -rf /usr/local/lib/python3.12/dist-packages/datasets*"
        )
    if not hasattr(module, "build_prepared_zsl_splits"):
        raise ImportError(
            f"Local datasets at {f} has no build_prepared_zsl_splits. "
            f"Check {prepared_py}."
        )
    return module
