#!/bin/bash
# =============================================================
# Benchmark HSPRN on all 5 ZSL datasets
# Usage: bash benchmark.sh /path/to/data/root
# =============================================================

set -e

DATA_ROOT="${1:?Usage: bash benchmark.sh /path/to/data/root}"
CONFIG_DIR="config"

echo "========================================"
echo " HSPRN Benchmark — All Datasets"
echo " Data root: ${DATA_ROOT}"
echo "========================================"

for dataset in awa1 awa2 cub sun apy; do
    echo ""
    echo "----------------------------------------"
    echo " Training: ${dataset^^}"
    echo "----------------------------------------"
    python train.py \
        --config "${CONFIG_DIR}/${dataset}.yaml" \
        --data_root "${DATA_ROOT}"
done

echo ""
echo "========================================"
echo " All benchmarks completed."
echo "========================================"
