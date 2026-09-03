#!/usr/bin/env bash
# Phase 2 cua v9: kiem tra ket luan lon nhat cua Phase 1 — ngan hang diem semantic
# (khong refine prototype) tot hon ngan hang refined tren ca hai thu nghiem:
#   +3.4 T1 tren trong so v8, +4.0 T1 va +14.9 u tren trong so R0.
# Phase 2 train truc tiep voi refiner tat (p_ref == p_sem) de loss va eval dung
# cung mot ngan hang diem, va doi chieu thuan R6a (φ grounded) vs R6b (φ plan5).
set -eu
cd /home/ubuntu/woker/Zeroshot/HSPRM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310
export CUDA_VISIBLE_DEVICES=0

RUNS="r6a_noref_grounded r6b_noref_v8phi"

echo "=== $(date '+%F %T') train $RUNS (80 epochs, seed 42, refiner OFF) ==="
for run in $RUNS; do
    echo "--- $(date '+%F %T') train $run ---"
    python train.py --config "config/cub_clip_v9_${run}_b16.yaml"
done

echo "=== $(date '+%F %T') eval: 9 cot, ca hai ngan hang diem (B4) ==="
for run in $RUNS; do
    cfg="config/cub_clip_v9_${run}_b16.yaml"
    dir="./checkpoints/cub_clip_v9_${run}_b16/CUB"
    for ckpt in best best_t1; do
        [ -f "$dir/$ckpt.pth" ] || continue
        for bank in refined semantic; do
            python tools/eval_checkpoint.py --config "$cfg" \
                --checkpoint "$dir/$ckpt.pth" \
                --name "${run}/${ckpt}/${bank}" \
                --score_bank "$bank"
        done
    done
done

echo "=== $(date '+%F %T') PHASE 2 DONE ==="
