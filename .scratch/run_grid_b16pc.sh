#!/bin/bash
# Sequential lambda-grid batch for HSPRN v8 B/16 precomputed (runs on IBM, GPU 0).
# Order: most-promising seen-bias counteractions first (lambda_sem/lambda_trans up),
# then the rest of the one-at-a-time sensitivity. Center = (lambda3=0.5, lambda_sem=1.0, lambda_trans=0.5).
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate py310
cd /home/ubuntu/woker/Zeroshot/HSPRM
export CUDA_VISIBLE_DEVICES=0

RUNS="grid_b16pc_lsem_200 grid_b16pc_ltrans_100 grid_b16pc_lsem_050 grid_b16pc_ltrans_025 grid_b16pc_l3_100 grid_b16pc_l3_025"

for cfg in $RUNS; do
  echo "===== START $cfg $(date -Is) ====="
  python train.py --config config/${cfg}.yaml > logs/${cfg}.log 2>&1
  echo "===== END   $cfg exit=$? $(date -Is) ====="
done
echo "===== ALL_GRID_DONE $(date -Is) ====="
