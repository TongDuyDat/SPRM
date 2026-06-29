#!/bin/bash
# ==============================================================================
# Script to download raw images for AWA2, CUB, APY, SUN with correct official URLs
# ==============================================================================

set -e

DATA_DIR="./data"
mkdir -p "${DATA_DIR}"

# 1. Animals with Attributes 2 (AWA2) ~ 13 GB
if [ -d "${DATA_DIR}/AWA2/JPEGImages" ]; then
    echo "AWA2 images already exist. Skipping."
else
    echo "=== Downloading AWA2 Images (~13 GB) ==="
    wget -c "https://cvml.ist.ac.at/AwA2/AwA2-data.zip"
    echo "Extracting AWA2..."
    unzip -q -o AwA2-data.zip -d "${DATA_DIR}/AWA2"
    # AwA2-data.zip extracts a JPEGImages directory directly inside the target
    rm -f AwA2-data.zip
fi

# 2. CUB-200-2011 ~ 1.1 GB
if [ -d "${DATA_DIR}/CUB/images" ]; then
    echo "CUB images already exist. Skipping."
else
    echo "=== Downloading CUB Images (~1.1 GB) ==="
    wget -c "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    echo "Extracting CUB..."
    tar -xzf CUB_200_2011.tgz -C "${DATA_DIR}/CUB"
    # Moving images folder from CUB_200_2011 to CUB root
    if [ -d "${DATA_DIR}/CUB/CUB_200_2011/images" ]; then
        mv "${DATA_DIR}/CUB/CUB_200_2011/images" "${DATA_DIR}/CUB/"
        rm -rf "${DATA_DIR}/CUB/CUB_200_2011"
    fi
    rm -f CUB_200_2011.tgz
fi

# 3. APY (Pascal Yahoo) ~ 1.5 GB
if [ -d "${DATA_DIR}/APY/images" ]; then
    echo "APY images already exist. Skipping."
else
    echo "=== Downloading APY Images (VOC2008 + Yahoo) ==="
    mkdir -p "${DATA_DIR}/APY/images"
    
    # Download Yahoo Images (approx 150 MB)
    wget -c "http://pascal.inrialpes.fr/data2/attribute/yahoo_images.tar"
    echo "Extracting Yahoo images..."
    tar -xf yahoo_images.tar -C "${DATA_DIR}/APY/images/"
    rm -f yahoo_images.tar
    
    # Download Pascal VOC 2008 Images (approx 1.3 GB)
    wget -c "http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar"
    echo "Extracting Pascal VOC 2008 images..."
    tar -xf VOCtrainval_14-Jul-2008.tar -C "${DATA_DIR}/APY/images/"
    rm -f VOCtrainval_14-Jul-2008.tar
fi

# 4. SUN Attributes ~ 3 GB
if [ -d "${DATA_DIR}/SUN/images" ]; then
    echo "SUN images already exist. Skipping."
else
    echo "=== Downloading SUN Images (~3 GB) ==="
    wget -c "https://cs.brown.edu/~gmpatter/source/sunattribute_images.tar.gz"
    echo "Extracting SUN..."
    tar -xzf sunattribute_images.tar.gz -C "${DATA_DIR}/SUN"
    rm -f sunattribute_images.tar.gz
fi

echo "================================================================================"
echo "All raw image datasets download and extraction complete!"
echo "Datasets directories:"
ls -F "${DATA_DIR}"
echo "================================================================================"
