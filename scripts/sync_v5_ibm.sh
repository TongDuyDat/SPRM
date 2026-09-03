#!/bin/bash
# Sync HSPRN-A changed files WSL -> IBM, then fix CRLF on the server side.
# Run from WSL:  bash scripts/sync_v5_ibm.sh
set -e
SCPEXE="/mnt/c/Windows/System32/OpenSSH/scp.exe"
SSHEXE="/mnt/c/Windows/System32/OpenSSH/ssh.exe"
KEY='D:\Work\ssh\tongdat_rsa.prv'
HOST="ubuntu@128.168.135.246"
REMOTE_DIR="/home/ubuntu/woker/Zeroshot/HSPRM"

FILES=(
  "model/aggregator.py"
  "model/sematics/constructor.py"
  "losses/hsprn_loss.py"
  "tools/evaluator.py"
  "trainer/trainer.py"
  "config/cub_clip_v5_r0_probe.yaml"
  "config/cub_clip_v5_r1_wv.yaml"
  "config/cub_clip_v5_r2_full.yaml"
  "config/cub_r101_v5_rb_bert.yaml"
  "scripts/run_v5_batch.sh"
)

cd "$(dirname "$0")/.."

for f in "${FILES[@]}"; do
  echo ">>> $f"
  "$SCPEXE" -o BatchMode=yes -o IdentitiesOnly=yes -i "$KEY" "$f" "$HOST:$REMOTE_DIR/$f"
done

# Strip CRLF (local files may be CRLF) and make the batch script executable.
REMOTE_FILES=$(printf '%s ' "${FILES[@]}")
"$SSHEXE" -o BatchMode=yes -o IdentitiesOnly=yes -i "$KEY" "$HOST" \
  "cd $REMOTE_DIR && sed -i 's/\r$//' $REMOTE_FILES && chmod +x scripts/run_v5_batch.sh && echo SYNC_OK"
