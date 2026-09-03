#!/bin/bash
# Detached sentinel: wait for the whole grid batch to finish, then write one
# final summary file (base + all 6 grid runs) for trivial collection.
cd /home/ubuntu/woker/Zeroshot/HSPRM || exit 1
while ! grep -q "ALL_GRID_DONE" logs/grid_b16pc_master.log 2>/dev/null; do
  sleep 60
done
sleep 10
{
  echo "=== GRID FINAL SUMMARY written $(date -Is) ==="
  bash collect_results_b16pc.sh 2>/dev/null
} > logs/grid_b16pc_FINAL_SUMMARY.txt 2>&1
