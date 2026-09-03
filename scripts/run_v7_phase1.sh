#!/bin/bash
# HSPRN-A v7 Phase 1: ceiling measurements.
# 1) Seen-class probes (linear + MLP) on B/32, B/16, L/14 frozen features.
# 2) R0 geometry probes on B/16 and L/14 (u/cal_u/ZSL of the frozen space).
# Launch: nohup bash scripts/run_v7_phase1.sh > logs/cub_clip_v7_phase1.log 2>&1 &
set -u
cd /home/ubuntu/woker/Zeroshot/HSPRM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310
export CUDA_VISIBLE_DEVICES=0

while pgrep -f "python train.py" > /dev/null || pgrep -f "probe_seen" > /dev/null; do
  echo "$(date '+%F %T') waiting for GPU..."
  sleep 60
done

echo "=== $(date '+%F %T') PHASE 1 START ==="

for bb in clip_vit_b32 clip_vit_b16 clip_vit_l14; do
  for pr in linear mlp; do
    echo "--- $(date '+%F %T') probe $bb / $pr ---"
    python tools/probe_seen.py --backbone "$bb" --probe "$pr" --epochs 15
  done
done

echo "--- $(date '+%F %T') R0 probe B/16 ---"
python train.py --config config/cub_clip_v7_r0_b16.yaml
echo "--- $(date '+%F %T') R0 probe L/14 ---"
python train.py --config config/cub_clip_v7_r0_l14.yaml

echo "=== $(date '+%F %T') PHASE 1 DONE ==="
