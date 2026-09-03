#!/usr/bin/env bash
# v9.6 Phase 4 — HUAN LUYEN 3 BO APY/AWA2/SUN voi semantics plan5 day du 4 truong
# (attr_c moi sinh theo chuan CUB), pattern R6b CUB: refiner OFF, semantic bank.
# Chuoi: refresh metadata -> precompute phi -> train 80ep -> eval 9 cot.
# Log: logs/v96_3ds_train.log. Chay tren IBM.
set -eu
cd /home/ubuntu/woker/Zeroshot/HSPRM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310
export CUDA_VISIBLE_DEVICES=0

for DS in APY AWA2 SUN; do
    ds=$(echo $DS | tr 'A-Z' 'a-z')
    CFG_DESC=config/${ds}_clip_v96_hsprn_b16.yaml
    CFG_TRAIN=config/${ds}_clip_v96_r6_plan5_b16.yaml
    CKPT=./checkpoints/${ds}_clip_v96_r6_plan5_b16/$DS

    echo "=== $(date '+%F %T') $DS: refresh metadata ==="
    python scripts/refresh_semar_3ds.py $DS

    echo "=== $(date '+%F %T') $DS: precompute phi ==="
    python tools/precompute_class_phi.py --config $CFG_DESC \
        --prepared_root ./prepared_datasets \
        --output ./pre-trained/${ds}_v96_phi_b16.pt

    echo "=== $(date '+%F %T') $DS: train R6-style 80ep ==="
    python train.py --config $CFG_TRAIN

    echo "=== $(date '+%F %T') $DS: eval final + best (semantic bank) ==="
    for ckpt in final best best_t1; do
        [ -f "$CKPT/$ckpt.pth" ] || continue
        python tools/eval_checkpoint.py --config $CFG_TRAIN \
            --checkpoint "$CKPT/$ckpt.pth" \
            --name "V96/${ds}/${ckpt}/semantic" --score_bank semantic
    done
done

echo "=== $(date '+%F %T') PHASE 4 (3 datasets) DONE ==="
