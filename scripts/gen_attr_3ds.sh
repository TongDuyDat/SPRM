#!/usr/bin/env bash
# v9.6: sinh attr_c cho APY/AWA2/SUN theo dung prompt attr_c cua chuan CUB
# (PROMPTS['attr_c'] trong tools/generate_descriptions.py), roi ghep voi
# desc_c/hier_c da co san trong file plan5 cua tung bo -> FILE MOI
# data_semantics/<DS>/class_descriptions_<ds>_after_llm_plan5.json
# (dung 4 truong nhu CUB: name_c, attr_c, hier_c, desc_c).
# KHONG de file plan5 cu. Log: logs/attr_gen_3ds.log
set -eu
cd /home/tongdat/NCKH/NCKH2026/ZeroShot/HSPRN

echo "=== $(date '+%F %T') START attr_c generation for APY AWA2 SUN ==="
for DS in APY AWA2 SUN; do
    echo "=== $(date '+%F %T') $DS: generate attr_c ==="
    python - <<PY
import json, sys, time
sys.path.insert(0, ".")
from tools import generate_descriptions as gen
from tools.clip_token_budget import filter_items_by_budget

ds = "$DS"
plan5_path = f"data_semantics/{ds}/class_descriptions_{ds.lower()}_plan5.json"
out_path = f"data_semantics/{ds}/class_descriptions_{ds.lower()}_after_llm_plan5.json"

plan5 = json.load(open(plan5_path))
class Args:
    api_base = gen.DEFAULT_API_BASE
    api_key = gen.DEFAULT_API_KEY
    model = gen.DEFAULT_MODEL
    max_tokens = gen.DEFAULT_MAX_TOKENS
    temperature = gen.DEFAULT_TEMPERATURE
    retries = 5
    backoff = 4.0
    timeout = 300.0
args = Args()

out = {}
failed = []
# Ky tu dac biet trong ten lop -> dung PROMPT_CLASS_ALIASES nhu generator chuan
for name, rec in plan5.items():
    prompt_name = gen.PROMPT_CLASS_ALIASES.get(name.lower(), name)
    prompt = gen.PROMPTS["attr_c"].format(class_name=prompt_name)
    try:
        raw = gen.call_with_retry(args, prompt)
        items = gen.parse_items(raw)
        items = filter_items_by_budget(items, max_tokens=77, on_overflow="skip_item")
        if len(items) < 5:
            raise ValueError(f"only {len(items)} items after filter")
    except Exception as e:
        failed.append(name)
        print(f"[FAIL] {name}: {e}", flush=True)
        continue
    out[name] = {
        "name_c": rec["name_c"],
        "attr_c": items[:10],
        "hier_c": rec["hier_c"],
        "desc_c": rec["desc_c"],
    }
    print(f"[OK] {name}: {len(items)} attr items", flush=True)
    # ghi tich luy sau moi lop de resume duoc neu do treo
    json.dump(out, open(out_path + ".partial", "w"), ensure_ascii=False, indent=1)

if failed:
    print(f"FAILED classes ({len(failed)}): {failed}")
    sys.exit(1)
json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)
import os
os.remove(out_path + ".partial")
print(f"=== {ds}: saved {len(out)}/{len(plan5)} classes -> {out_path}")
PY
done
echo "=== $(date '+%F %T') ALL DONE ==="
