#!/usr/bin/env bash
# One-shot: sync all v10 files to IBM, fix CRLF, precompute 4 attrq banks, run E0 sanity.
# Run from repo root on WSL:  bash .scratch/v10_sync_and_e0.sh
set -e

SCP="/mnt/c/Windows/System32/OpenSSH/scp.exe -o BatchMode=yes -o IdentitiesOnly=yes -i D:\\Work\\ssh\\tongdat_rsa.prv"
IBM="ubuntu@128.168.135.246"
ROOT="/home/ubuntu/woker/Zeroshot/HSPRM"

echo "=== [1/3] Sync model/ core ==="
$SCP model/vision_model.py model/aggregator.py model/protypes.py model/hsprn_model.py \
    model/hsprn_outputs.py model/sematics/constructor.py "$IBM:$ROOT/model/"
$SCP model/vision/__init__.py "$IBM:$ROOT/model/vision/"

echo "=== [1/3] Sync losses/ trainer/ tools/ ==="
$SCP losses/hsprn_loss.py "$IBM:$ROOT/losses/"
$SCP trainer/trainer.py "$IBM:$ROOT/trainer/"
$SCP tools/semantic_policy.py tools/precompute_attribute_queries.py tools/e0_sanity_v10.py "$IBM:$ROOT/tools/"

echo "=== [1/3] Sync 12 v10 configs ==="
$SCP config/cub_clip_v10_ca_b16.yaml config/cub_clip_v10_a2_mfacls.yaml config/cub_clip_v10_a3_noattn.yaml \
    config/apy_clip_v10_ca_b16.yaml config/apy_clip_v10_a2_mfacls.yaml config/apy_clip_v10_a3_noattn.yaml \
    config/awa2_clip_v10_ca_b16.yaml config/awa2_clip_v10_a2_mfacls.yaml config/awa2_clip_v10_a3_noattn.yaml \
    config/sun_clip_v10_ca_b16.yaml config/sun_clip_v10_a2_mfacls.yaml config/sun_clip_v10_a3_noattn.yaml \
    "$IBM:$ROOT/config/"

echo "=== [2/3] CRLF fix on IBM ==="
bash ~/ssh_ibm.sh "cd $ROOT && sed -i 's/\r\$//' \
    model/vision_model.py model/aggregator.py model/protypes.py model/hsprn_model.py \
    model/hsprn_outputs.py model/sematics/constructor.py model/vision/__init__.py \
    losses/hsprn_loss.py trainer/trainer.py \
    tools/semantic_policy.py tools/precompute_attribute_queries.py tools/e0_sanity_v10.py \
    config/*v10*.yaml && echo CRLF_OK"

echo "=== [3/3] Precompute 4 attribute query banks + run E0 sanity ==="
bash ~/ssh_ibm.sh "cd $ROOT && source ~/miniconda3/etc/profile.d/conda.sh && conda activate py310 && \
python tools/precompute_attribute_queries.py --config config/cub_clip_v10_ca_b16.yaml --output pre-trained/cub_attrq_b16.pt && \
python tools/precompute_attribute_queries.py --config config/apy_clip_v10_ca_b16.yaml --output pre-trained/apy_attrq_b16.pt && \
python tools/precompute_attribute_queries.py --config config/awa2_clip_v10_ca_b16.yaml --output pre-trained/awa2_attrq_b16.pt && \
python tools/precompute_attribute_queries.py --config config/sun_clip_v10_ca_b16.yaml --output pre-trained/sun_attrq_b16.pt && \
echo '=== E0 SANITY ===' && \
python tools/e0_sanity_v10.py --baseline config/cub_clip_v9_r6b_noref_v8phi_b16.yaml --v10 config/cub_clip_v10_ca_b16.yaml --device cuda"

echo "=== DONE: exit code above tells E0 verdict ==="
