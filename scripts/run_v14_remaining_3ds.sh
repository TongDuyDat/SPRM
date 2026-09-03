#!/usr/bin/env bash
# Runner script: Train & Eval HSPRN-v14 Super-CLIP on remaining 3 datasets (AWA2, APY, SUN)
# Protocol: 50 Epochs, Batch Size 512, Phased Training (3 Phases)

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310
cd /home/ubuntu/woker/Zeroshot/HSPRM

DATASETS=("awa2" "apy" "sun")

for ds in "${DATASETS[@]}"; do
    CFG="config/${ds}_clip_v14_super_50ep.yaml"
    LOG="logs/v14_super_${ds}.log"
    CKPT_DIR="checkpoints/${ds}_clip_v14_super_50ep"

    echo "========================================================================"
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] BẮT ĐẦU HUẤN LUYỆN v14 SUPER-CLIP TRÊN $ds ==="
    echo "========================================================================"

    rm -rf "$CKPT_DIR" "$LOG" "logs/${ds}_clip_v14_super_50ep"
    CUDA_VISIBLE_DEVICES=0 python train.py --config "$CFG" > "$LOG" 2>&1

    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] HOÀN THÀNH HUẤN LUYỆN $ds -> TIẾN HÀNH ĐÁNH GIÁ ==="
    
    for ckpt_name in "best" "best_t1" "final"; do
        CKPT_FILE="${CKPT_DIR}/${ds^^}/${ckpt_name}.pth"
        if [ -f "$CKPT_FILE" ]; then
            for bank in "refined" "semantic"; do
                echo "--- Eval $ds $ckpt_name ($bank) ---"
                CUDA_VISIBLE_DEVICES=0 python tools/eval_checkpoint.py --config "$CFG" --checkpoint "$CKPT_FILE" --score_bank "$bank" 2>&1 | grep RESULT || true
            done
        fi
    done
    echo ""
done

echo "========================================================================"
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] TẤT CẢ 3 BỘ DỮ LIỆU ĐÃ HOÀN THÀNH XONG! ==="
echo "========================================================================"
