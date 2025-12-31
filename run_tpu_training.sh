#!/bin/bash

# TPU Training Script for Gemma2 2B
# Usage: ./run_tpu_training.sh

set -e

echo "==================================="
echo "Gemma2 2B TPU Training"
echo "==================================="

# Check if running on TPU VM
if [ -z "$TPU_NAME" ]; then
    echo "Warning: TPU_NAME not set. Make sure you're running on a TPU VM."
fi

# Create output directory
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs"}
mkdir -p "$OUTPUT_DIR"

# Set environment variables for TPU
export XLA_USE_BF16=1
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000
export PJRT_DEVICE=TPU

# Model configuration
MODEL_NAME=${MODEL_NAME:-"google/gemma-2-2b"}
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
MAX_LENGTH=${MAX_LENGTH:-2048}
NUM_TPU_CORES=${NUM_TPU_CORES:-8}

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Max Length: $MAX_LENGTH"
echo "  TPU Cores: $NUM_TPU_CORES"
echo ""

# Run training
python3 train.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --max_length "$MAX_LENGTH" \
    --num_tpu_cores "$NUM_TPU_CORES" \
    --warmup_steps 100 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 100 \
    --seed 42

echo ""
echo "==================================="
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "==================================="
