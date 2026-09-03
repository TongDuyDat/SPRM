#!/bin/bash
# HSPRN v9 Phase 1 (chạy trên IBM): semantics grounded -> cổng G1/G2 -> smoke G3
# -> train R0..R4 -> eval đủ 9 cột cho mỗi biến thể.
#
# Khởi động:
#   nohup bash scripts/run_v9_phase1.sh > logs/cub_clip_v9_phase1.log 2>&1 &
#
# Điều kiện tiên quyết (làm ở máy local, KHÔNG train local):
#   1) data_semantics/CUB/class_descriptions_cub_grounded_after_llm.json đã xong
#      và in "ĐẠT G1" khi chạy tools/check_grounded_semantics.py.
#   2) Refresh metadata của prepared_datasets (chạy local, chỉ ghi JSON):
#        python tools/prepare_zsl_dataset_package.py --action manifest --datasets CUB \
#            --overwrite --require-full-semantics \
#            --semantic-metadata-template "./data_semantics/{dataset}/class_descriptions_{dataset_lower}_grounded_after_llm.json"
#      (chế độ data.source=prepared nạp prepared_datasets/CUB/metadata/class_semantics.json,
#       không đọc thẳng data_semantics, nên bước này bắt buộc.)
#   3) Copy sang IBM: file JSON ở (1), prepared_datasets/CUB/metadata/ ở (2), 6 config
#      config/cub_clip_v9_*.yaml, và các tool đã sửa.
#
# Cổng: G1 = hình thái văn bản; G2a = cosine giữa các lớp từng field (desc_c phải
# <= 0.90, trước đây 0.95); G2b = dòng "mean off-diagonal cosine: X -> Y" của
# precompute (Y sau center phải <= 0.5); G3 = smoke 2 epoch không NaN/crash.
set -eu
cd /home/ubuntu/woker/Zeroshot/HSPRM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310
export CUDA_VISIBLE_DEVICES=0

GROUNDED=./data_semantics/CUB/class_descriptions_cub_grounded_after_llm.json
OLD=./data_semantics/CUB/class_descriptions_cub_after_llm_plan5.json
CFG_CLASSDESC=./config/cub_clip_v9_hsprn_b16.yaml
CLIP=./pre-trained/ViT-B-16.pt
RUNS="r0_grounded r1_fnorm r2_center r2b_alignall r3_residual r4_fieldw"
# R2b dùng chung file phi với R2: chỉ thêm loss.align_all_classes, không có B3.
# Nhờ vậy R3 - R2b = đóng góp thật của B3, R2b - R2 = đóng góp của chính align_all.

echo "=== $(date '+%F %T') G1: kiểm tra hình thái semantics grounded ==="
python tools/check_grounded_semantics.py --semantics "$GROUNDED" --baseline "$OLD"

echo "=== $(date '+%F %T') G2a: do phân biệt từng field (grounded vs cũ, ViT-B/16) ==="
python scripts/analyze_semantic_fields.py --semantics "$GROUNDED" --clip_model "$CLIP" \
    | tee logs/v9_g2a_grounded.log
python scripts/analyze_semantic_fields.py --semantics "$OLD" --clip_model "$CLIP" \
    | tee logs/v9_g2a_old.log

echo "=== $(date '+%F %T') G2b: precompute phi cho 4 biến thể ==="
python tools/precompute_class_phi.py --config "$CFG_CLASSDESC" \
    --prepared_root ./prepared_datasets --output ./pre-trained/cub_v9_r0_phi_b16.pt
python tools/precompute_class_phi.py --config "$CFG_CLASSDESC" --field_norm \
    --prepared_root ./prepared_datasets --output ./pre-trained/cub_v9_r1_phi_b16.pt
python tools/precompute_class_phi.py --config "$CFG_CLASSDESC" --field_norm --center \
    --prepared_root ./prepared_datasets --output ./pre-trained/cub_v9_r2_phi_b16.pt
python tools/precompute_class_phi.py --config "$CFG_CLASSDESC" --field_norm --center \
    --field_weights 3,0,1,1 --prepared_root ./prepared_datasets \
    --output ./pre-trained/cub_v9_r4_phi_b16.pt

echo "=== $(date '+%F %T') G3: smoke 2 epochs trên config nặng nhất (R3) ==="
python train.py --config config/cub_clip_v9_r3_residual_b16.yaml \
    --epochs 2 --subset_size 2048 --eval_subset_size 1024 --num_workers 8

echo "=== $(date '+%F %T') train R0..R4 (80 epochs, seed 42) ==="
for run in $RUNS; do
    echo "--- $(date '+%F %T') train $run ---"
    python train.py --config "config/cub_clip_v9_${run}_b16.yaml"
done

echo "=== $(date '+%F %T') eval: 9 cột, cả hai ngân hàng điểm (B4) ==="
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

echo "=== $(date '+%F %T') PHASE 1 DONE ==="
