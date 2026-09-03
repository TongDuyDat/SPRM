#!/bin/bash
# Watcher: wait for the base precomputed training to finish, then run the grid batch.
# Launched once via nohup; makes the base->grid handoff fully automatic.
set -u
cd /home/ubuntu/woker/Zeroshot/HSPRM
echo "watcher started $(date -Is); waiting for base to finish"
while true; do
  if ! ps aux | grep "cub_clip_v8_hsprn_b16_precomputed" | grep "python train.py" | grep -v grep >/dev/null; then
    sleep 5  # guard against a transient ps miss
    if ! ps aux | grep "cub_clip_v8_hsprn_b16_precomputed" | grep "python train.py" | grep -v grep >/dev/null; then
      echo "base confirmed finished at $(date -Is)"
      break
    fi
  fi
  sleep 30
done
echo "launching grid batch $(date -Is)"
bash run_grid_b16pc.sh > logs/grid_b16pc_master.log 2>&1
echo "grid batch completed $(date -Is)"
