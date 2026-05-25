#!/usr/bin/env bash
# Download and organise the BUSI dataset.
#
# Two methods are supported; the script tries Method 1 first:
#   Method 1 (recommended): Kaggle CLI  — fastest, most reliable
#   Method 2 (fallback):    Direct wget — no API key needed
#
# Usage:
#   bash scripts/download_data.sh            # Kaggle method
#   bash scripts/download_data.sh --wget     # direct download method

set -euo pipefail

DATA_DIR="data/BUSI"
KAGGLE_DATASET="aryashah2k/breast-ultrasound-images-dataset"
DIRECT_URL="https://data.mendeley.com/public-files/datasets/wmy84gzngw/files/\
f7c739d9-bae3-4f11-bd97-bd26ed8e7c6b/file_downloaded"

USE_WGET=false
[[ "${1:-}" == "--wget" ]] && USE_WGET=true

if [[ -d "$DATA_DIR/benign" && -d "$DATA_DIR/malignant" ]]; then
    echo "BUSI data already present at $DATA_DIR — skipping download."
    exit 0
fi

mkdir -p "$DATA_DIR"

if [[ "$USE_WGET" == false ]]; then
    # ── Method 1: Kaggle CLI ──────────────────────────────────────────────
    if ! command -v kaggle &>/dev/null; then
        echo "[ERROR] kaggle CLI not found. Install with: pip install kaggle"
        echo "        Then place ~/.kaggle/kaggle.json with your API credentials."
        echo "        Or run: bash scripts/download_data.sh --wget"
        exit 1
    fi
    echo "Downloading BUSI via Kaggle CLI..."
    kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_DIR" --unzip
    # Kaggle extracts to Dataset_BUSI_with_GT/; move to expected layout
    if [[ -d "$DATA_DIR/Dataset_BUSI_with_GT" ]]; then
        mv "$DATA_DIR/Dataset_BUSI_with_GT"/* "$DATA_DIR/"
        rmdir "$DATA_DIR/Dataset_BUSI_with_GT"
    fi
else
    # ── Method 2: Direct download from Mendeley Data ─────────────────────
    echo "Downloading BUSI from Mendeley Data (may take a few minutes)..."
    wget -q --show-progress -O "$DATA_DIR/busi.zip" "$DIRECT_URL"
    unzip -q "$DATA_DIR/busi.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/busi.zip"
    # Move nested directory if present
    NESTED=$(find "$DATA_DIR" -maxdepth 1 -mindepth 1 -type d | head -n 1)
    if [[ -n "$NESTED" && "$NESTED" != "$DATA_DIR" ]]; then
        mv "$NESTED"/* "$DATA_DIR/"
        rmdir "$NESTED"
    fi
fi

# Validate expected structure
for cls in benign malignant normal; do
    count=$(find "$DATA_DIR/$cls" -name "*.png" ! -name "*_mask*" 2>/dev/null | wc -l)
    echo "  $cls: $count images"
done

echo "Done. Dataset ready at $DATA_DIR"
