#!/bin/bash
# Collect best-epoch results for the base run + all grid variants.
# For each: print the final "Done. Best Cal H" summary and the last *BEST* epoch line
# (which carries raw u/s/H, ZSL T1, Cal u/s/H, gamma).
set -u
cd /home/ubuntu/woker/Zeroshot/HSPRM

show() {
  local label="$1" L="$2"
  echo "================ $label ================"
  if [ -f "$L" ]; then
    grep -E "Done. Best Cal H" "$L" | tail -1
    grep -E "INFO - Epoch" "$L" | grep '\*BEST\*' | tail -1 | sed 's/.*INFO - Epoch/Epoch/'
    if ! grep -qE "Done. Best Cal H" "$L"; then echo "(still running: no Done line yet)"; fi
  else
    echo "(log not present yet)"
  fi
}

show "BASE (l3=0.5 lsem=1.0 ltrans=0.5)" logs/v8_b16_precomputed.log
show "GRID lsem=2.0"  logs/grid_b16pc_lsem_200.log
show "GRID ltrans=1.0" logs/grid_b16pc_ltrans_100.log
show "GRID lsem=0.5"  logs/grid_b16pc_lsem_050.log
show "GRID ltrans=0.25" logs/grid_b16pc_ltrans_025.log
show "GRID l3=1.0"    logs/grid_b16pc_l3_100.log
show "GRID l3=0.25"   logs/grid_b16pc_l3_025.log
echo "================ master ================"
[ -f logs/grid_b16pc_master.log ] && cat logs/grid_b16pc_master.log || echo "(master log not present yet)"
