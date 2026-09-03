#!/bin/bash
# HSPRN-A v5.2 batch (R3a -> R3b -> R3c), run on the IBM server from the repo
# root:  nohup bash scripts/run_v5_r3_batch.sh > logs/cub_clip_v5_r3_batch.log 2>&1 &
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

run cub_clip_v5_r3a_noIPR.yaml   R3a_noIPR
run cub_clip_v5_r3b_soft.yaml    R3b_soft
run cub_clip_v5_r3c_trans10.yaml R3c_trans10

echo "=== $(date '+%F %T') R3 BATCH DONE ==="
