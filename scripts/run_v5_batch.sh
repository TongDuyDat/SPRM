#!/bin/bash
# HSPRN-A batch (R0 probe -> R1 -> R2), run on the IBM server from the repo
# root:  nohup bash scripts/run_v5_batch.sh > logs/cub_clip_v5_batch.log 2>&1 &
set -u
cd /home/ubuntu/woker/Zeroshot/HSPRM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310
export CUDA_VISIBLE_DEVICES=0

run() {
  local cfg="$1"
  local name="$2"
  echo "=== $(date '+%F %T') START $name ==="
  python train.py --config "config/$cfg"
  echo "=== $(date '+%F %T') END $name (exit=$?) ==="
}

run cub_clip_v5_r0_probe.yaml R0_probe
run cub_clip_v5_r1_wv.yaml    R1_wv
run cub_clip_v5_r2_full.yaml  R2_full

echo "=== $(date '+%F %T') BATCH DONE ==="
