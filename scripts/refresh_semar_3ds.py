#!/usr/bin/env python
"""v9.6: refresh prepared_datasets/<DS>/metadata/class_semantics.json cho
APY/AWA2/SUN tu file semantics moi (4 truong, attr_c list 10 cau chuan CUB).

Backup ban cu thanh class_semantics.json.bak_v96 truoc khi ghi.
Output la LIST theo thu tu class_index (dang ma PreparedZSLDataset doc),
xay tu dict {class_name: record} bang thu tu class_names cua classes.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def refresh(ds):
    ds = ds.upper()
    meta_dir = ROOT / "prepared_datasets" / ds / "metadata"
    classes = json.load(open(meta_dir / "classes.json"))
    class_names = [c["class_name"] for c in sorted(classes, key=lambda x: int(x["class_index"]))]

    new = json.load(open(ROOT / "data_semantics" / ds / f"class_descriptions_{ds.lower()}_after_llm_plan5.json"))
    missing = [n for n in class_names if n not in new]
    if missing:
        raise SystemExit(f"{ds}: classes in dataset but not in new semantics: {missing[:5]}")

    # list theo thu tu class_index
    records = [new[n] for n in class_names]
    # verify 4 truong
    for i, r in enumerate(records):
        for f in ("name_c", "attr_c", "hier_c", "desc_c"):
            if not r.get(f):
                raise SystemExit(f"{ds}: record #{i} ({class_names[i]}) missing {f}")

    target = meta_dir / "class_semantics.json"
    backup = meta_dir / "class_semantics.json.bak_v96"
    if target.exists() and not backup.exists():
        backup.write_text(target.read_text())
    target.write_text(json.dumps(records, ensure_ascii=False, indent=1))
    print(f"{ds}: refreshed {len(records)} classes -> {target} (backup: {backup.name})")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["APY", "AWA2", "SUN"]):
        refresh(ds)
