#!/bin/bash

# If the first argument is a flag (starts with --), use defaults for positional arguments
if [[ "$1" == --* ]]; then
    echo "Detected flags. Using default Research Configuration (MNIST-IDX/Amplitude)..."
    ./.venv/bin/python main.py \
        --dataset "idx" \
        --path "datasets/MNIST/" \
        --samples 500 \
        --encoding "amplitude" \
        --learning-rate 0.005 \
        --summary-log "training_summary.txt" \
        --no-profile \
        "$@"
    exit $?
fi

# 1. Dataset Type (auto, synthetic, npz, csv, mnist, idx)
DATASET_TYPE="${1:-idx}"

# 2. Dataset Path
DATASET_PATH="${2:-datasets/MNIST/}"

# 3. Encoding Strategy (auto, patch, amplitude, feature_map)
ENCODING="${3:-amplitude}"

# 4. Image Size
IMAGE_SIZE="${4:-28}"

# 5. Samples
SAMPLES="${5:-500}"

# 6. Learning Rate
LEARNING_RATE="${6:-0.005}"

# Run main python script using the project virtual environment
echo "------------------------------------------------"
echo "Quantum Native QCNN - Research Accuracy Launcher"
echo "------------------------------------------------"
echo "Dataset: $DATASET_TYPE"
echo "Encoding: $ENCODING (Size: $IMAGE_SIZE)"
echo "Samples: $SAMPLES | LR: $LEARNING_RATE"
echo "------------------------------------------------"

./.venv/bin/python main.py \
    --dataset "$DATASET_TYPE" \
    --path "$DATASET_PATH" \
    --encoding "$ENCODING" \
    --image-size "$IMAGE_SIZE" \
    --samples "$SAMPLES" \
    --learning-rate "$LEARNING_RATE" \
    --summary-log "training_summary.txt" \
    --no-profile \
    "${@:7}"
