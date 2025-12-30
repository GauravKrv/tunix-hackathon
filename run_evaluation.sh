#!/bin/bash
# Example script for running model evaluation

# Set paths
BASE_MODEL="path/to/base/model"
FINETUNED_MODEL="path/to/finetuned/model"
OUTPUT_DIR="evaluation_results"

# Run evaluation
python evaluate.py \
    --base-model "$BASE_MODEL" \
    --finetuned-model "$FINETUNED_MODEL" \
    --benchmarks gsm8k math arc mmlu \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 4 \
    --num-samples 100 \
    --temperature 0.7 \
    --device cuda \
    --seed 42

echo "Evaluation complete! Results saved to: $OUTPUT_DIR"
echo "View HTML report at: $OUTPUT_DIR/comparison_report.html"
