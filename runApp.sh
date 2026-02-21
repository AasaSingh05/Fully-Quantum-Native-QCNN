#!/bin/bash

# If the first argument is a flag (starts with --), use defaults for positional arguments
if [[ "$1" == --* ]]; then
    echo "💡 Detected flags. Using default dataset (MNIST/auto)..."
    ./.venv/bin/python main.py \
        --dataset "auto" \
        --path "datasets/MNIST" \
        --encoding "auto" \
        --image-size "28" \
        --classes "0" "1" \
        --no-profile \
        "$@"
    exit $?
fi

# Configuration for QCNN Dataset Training
# Usage: ./runApp.sh [type] [path] [encoding] [image_size] [class1 class2]

# 1. Dataset Type (auto, idx, images, npz, csv, mnist)
DATASET_TYPE="${1:-auto}"

# 2. Dataset Path (auto-detects in datasets/ if relative)
DATASET_PATH="${2:-datasets/MNIST}"

# 3. Encoding Strategy (auto, patch, amplitude, feature_map)
ENCODING="${3:-auto}"

# 4. Image Size (Required for images/idx, automatically handled for others)
IMAGE_SIZE="${4:-28}"

# 5. Binary Classes (Defaults to 0 and 1 for MNIST/IDX)
CLASS_A="${5:-0}"
CLASS_B="${6:-1}"

# Run main python script using the project virtual environment
echo "------------------------------------------------"
echo "🚀 Quantum Native QCNN - Training Launcher"
echo "------------------------------------------------"
echo "📁 Dataset: $DATASET_PATH ($DATASET_TYPE)"
echo "🧬 Encoding: $ENCODING (Size: $IMAGE_SIZE)"
echo "🎯 Classes: $CLASS_A vs $CLASS_B"
echo "------------------------------------------------"

./.venv/bin/python main.py \
    --dataset "$DATASET_TYPE" \
    --path "$DATASET_PATH" \
    --encoding "$ENCODING" \
    --image-size "$IMAGE_SIZE" \
    --classes "$CLASS_A" "$CLASS_B" \
    --no-profile \
    "${@:7}"
