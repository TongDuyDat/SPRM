#!/usr/bin/env bash
# v9.6 Phase 3 — hop nhat 4 de xuat theo 3 tang (E1 -> E6), gate tung buoc.
# Tat ca chay tren IBM (L40S). Moi buoc in RESULT/TTA_RESULT 9 cot day du.
# Yeu cau audit: moi buoc phai copy nguyen van ket qua vao auditlog.md.
#
# Nhat ky smoke test da lam truoc khi chay that (xem auditlog E0):
#   - export_sentences: OK, tensor (200, 25, 512)
#   - control name_repeat/random_words: OK
#   - semantic_ens: OK sau khi sua thu tu class (bug -> T1 0.1 -> 72.3)
#   - CSLS: BI BAC BO (moi beta deu duoi baseline; beta=1 T1 29.0, beta=0.05
#     T1 73.8 < 74.4) — hub term vuot margin. E4 chi chay 3 hang lam bang
#     chung cho auditlog, khong doi vao to hop.
set -eu
cd /home/ubuntu/woker/Zeroshot/HSPRM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310
export CUDA_VISIBLE_DEVICES=0

R6B=./checkpoints/cub_clip_v9_r6b_noref_v8phi_b16/CUB
R7H=./checkpoints/cub_clip_v9_r7h_hybrid_b16/CUB
P5_META=./data_semantics/CUB/class_descriptions_cub_after_llm_plan5.json
HYB_META=./prepared_datasets/CUB/metadata/class_descriptions_cub_hybrid.json
CFG_CLASSDESC=config/cub_clip_v9_hsprn_b16.yaml
CFG_R6B=config/cub_clip_v9_r6b_noref_v8phi_b16.yaml
CFG_R7H=config/cub_clip_v9_r7h_hybrid_b16.yaml

echo "=== $(date '+%F %T') E1: re-eval R6b ep44 baseline (semantic) + diagnostic ==="
python tools/eval_checkpoint.py --config $CFG_R6B --checkpoint $R6B/best.pth \
    --name "E1/r6b/ep44/semantic" --score_bank semantic

echo "=== $(date '+%F %T') E2a: precompute phi plan5 per-sentence ==="
python tools/precompute_class_phi.py --config $CFG_CLASSDESC \
    --prepared_root ./prepared_datasets --export_sentences \
    --metadata $P5_META --output ./pre-trained/cub_v9_plan5_sent_b16.pt
echo "=== $(date '+%F %T') E2b: Waffle controls (random_words / name_repeat) ==="
python tools/precompute_class_phi.py --config $CFG_CLASSDESC \
    --prepared_root ./prepared_datasets --control random_words \
    --output ./pre-trained/cub_v9_waffle_rand_b16.pt
python tools/precompute_class_phi.py --config $CFG_CLASSDESC \
    --prepared_root ./prepared_datasets --control name_repeat \
    --output ./pre-trained/cub_v9_waffle_name_b16.pt

echo "=== $(date '+%F %T') E2c: eval DESC-ENSEMBLE + controls tren R6b/ep44 ==="
python tools/eval_checkpoint.py --config $CFG_R6B --checkpoint $R6B/best.pth \
    --name "E2/r6b/ens_plan5" --score_bank semantic_ens \
    --phi_path ./pre-trained/cub_v9_plan5_sent_b16.pt
python tools/eval_checkpoint.py --config $CFG_R6B --checkpoint $R6B/best.pth \
    --name "E2/r6b/bank_rand" --score_bank semantic \
    --phi_path ./pre-trained/cub_v9_waffle_rand_b16.pt
python tools/eval_checkpoint.py --config $CFG_R6B --checkpoint $R6B/best.pth \
    --name "E2/r6b/bank_name" --score_bank semantic \
    --phi_path ./pre-trained/cub_v9_waffle_name_b16.pt
python tools/eval_checkpoint.py --config $CFG_R6B --checkpoint $R6B/best.pth \
    --name "E2/r6b/ens_rand" --score_bank semantic_ens \
    --phi_path ./pre-trained/cub_v9_waffle_rand_b16.pt
python tools/eval_checkpoint.py --config $CFG_R6B --checkpoint $R6B/best.pth \
    --name "E2/r6b/ens_name" --score_bank semantic_ens \
    --phi_path ./pre-trained/cub_v9_waffle_name_b16.pt

echo "=== $(date '+%F %T') E3a: precompute phi HYBRID (per-sentence kem theo) ==="
python tools/precompute_class_phi.py --config $CFG_CLASSDESC \
    --prepared_root ./prepared_datasets --export_sentences \
    --metadata $HYB_META --output ./pre-trained/cub_v9_hybrid_phi_b16.pt
echo "=== $(date '+%F %T') E3b: train R7h hybrid 80ep ==="
python train.py --config $CFG_R7H
echo "=== $(date '+%F %T') E3c: eval R7h final + best + ens ==="
python tools/eval_checkpoint.py --config $CFG_R7H --checkpoint $R7H/final.pth \
    --name "E3/r7h/final/semantic" --score_bank semantic
python tools/eval_checkpoint.py --config $CFG_R7H --checkpoint $R7H/final.pth \
    --name "E3/r7h/final/ens" --score_bank semantic_ens \
    --phi_path ./pre-trained/cub_v9_hybrid_phi_b16.pt
python tools/eval_checkpoint.py --config $CFG_R7H --checkpoint $R7H/best.pth \
    --name "E3/r7h/best/semantic" --score_bank semantic

echo "=== $(date '+%F %T') E4: CSLS bang chung bac bo (3 hang, khong doi vao to hop) ==="
python tools/eval_checkpoint.py --config $CFG_R6B --checkpoint $R6B/best.pth \
    --name "E4/r6b/csls_k10_b1.0" --score_bank semantic \
    --score_correction csls --csls_k 10 --csls_beta 1.0
python tools/eval_checkpoint.py --config $CFG_R6B --checkpoint $R6B/best.pth \
    --name "E4/r6b/csls_k10_b0.05" --score_bank semantic \
    --score_correction csls --csls_k 10 --csls_beta 0.05
python tools/eval_checkpoint.py --config $CFG_R6B --checkpoint $R6B/best.pth \
    --name "E4/r6b/csls_k20_b0.5" --score_bank semantic \
    --score_correction csls --csls_k 20 --csls_beta 0.5

echo "=== $(date '+%F %T') E5: TTA transductive (BANG RIENG) tren R6b ep44 ==="
for LAM in 0.3 0.5; do
    python tools/eval_tta.py --config $CFG_R6B --checkpoint $R6B/best.pth \
        --name "E5/r6b/tta_lam$LAM" --score_bank semantic \
        --lambda_mix $LAM --iters 10 --eval_iters 0,5,10
done

echo "=== $(date '+%F %T') PHASE 3 CORE DONE ==="
