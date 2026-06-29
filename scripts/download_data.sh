#!/bin/bash
# ==============================================================================
# Script to download ZSL benchmarks (AWA1, AWA2, CUB, SUN, APY) from xlsa17
# ==============================================================================

set -e

# Define data directory inside workspace
DATA_DIR="./data"
ZIP_FILE="xlsa17.zip"
URL="https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"

echo "Creating data directory at: ${DATA_DIR}"
mkdir -p "${DATA_DIR}"

# Check if zip file already exists
if [ -f "${ZIP_FILE}" ]; then
    echo "${ZIP_FILE} already exists. Skipping download."
else
    echo "Downloading xlsa17.zip (approx. 2.0 GB) from ${URL}..."
    # Using wget or curl depending on availability
    if command -v wget >/dev/null 2>&1; then
        wget -c "${URL}" -O "${ZIP_FILE}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L "${URL}" -o "${ZIP_FILE}"
    else
        echo "Error: Neither wget nor curl is installed. Please install one of them."
        exit 1
    fi
fi

echo "Extracting ${ZIP_FILE}..."
if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "${ZIP_FILE}"
    
    # Check if 'xlsa17/data' was extracted
    if [ -d "xlsa17/data" ]; then
        echo "Moving extracted datasets to ${DATA_DIR}..."
        mv xlsa17/data/* "${DATA_DIR}/"
        rm -rf xlsa17
    fi
else
    echo "Error: 'unzip' utility is not installed. Please install it (e.g. sudo apt install unzip)."
    exit 1
fi

# Clean up zip file if wanted (uncomment if you want to save disk space)
# rm "${ZIP_FILE}"

echo "================================================================================"
echo "Download and extraction complete!"
echo "Datasets directory structure:"
ls -F "${DATA_DIR}"
echo "================================================================================"
