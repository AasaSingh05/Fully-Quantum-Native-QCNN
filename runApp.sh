#!/bin/bash

# Configuration for custom MNIST dataset
DATASET_TYPE="${1:-images}"
DATASET_PATH="${2:-datasets/images/}"
ENCODING="${3:-patch}"
IMAGE_SIZE="${4:-28}"

# Run main python script using the project virtual environment
echo "Running QCNN Training with $DATASET_TYPE dataset from $DATASET_PATH..."
./.venv/bin/python main.py \
    --dataset "$DATASET_TYPE" \
    --path "$DATASET_PATH" \
    --encoding "$ENCODING" \
    --image-size "$IMAGE_SIZE" \
    --no-profile \
    "${@:5}"
