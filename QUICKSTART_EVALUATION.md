# Quick Start Guide for Model Evaluation

This guide will help you quickly get started with evaluating your trained models.

## Installation

1. Install dependencies:
```bash
pip install -r requirements-eval.txt
```

## Quick Evaluation

### 1. Evaluate a Single Model

```bash
python evaluate.py \
    --base-model "gpt2" \
    --benchmarks gsm8k math \
    --num-samples 10 \
    --output-dir quick_test
```

### 2. Compare Base vs Fine-tuned Model

```bash
python evaluate.py \
    --base-model "gpt2" \
    --finetuned-model "path/to/your/finetuned/model" \
    --benchmarks gsm8k math arc \
    --output-dir results
```

### 3. View Results

After evaluation completes, open the HTML report:
```bash
# On macOS
open results/comparison_report.html

# On Linux
xdg-open results/comparison_report.html

# On Windows
start results/comparison_report.html
```

Or view JSON results:
```bash
cat results/comparison_report.json | python -m json.tool
```

## Advanced Usage

### Batch Evaluation

1. Create a batch configuration file (see `batch_config_example.json`)
2. Run batch evaluation:

```bash
python batch_evaluate.py \
    --config batch_config_example.json \
    --output-dir batch_results
```

### Generate Visualizations

```bash
python visualize_results.py \
    --report results/comparison_report.json \
    --output-dir plots
```

## Directory Structure After Evaluation

```
evaluation_results/
├── base_gsm8k_results.json          # Base model results on GSM8K
├── finetuned_gsm8k_results.json     # Fine-tuned model results on GSM8K
├── base_math_results.json            # Base model results on MATH
├── finetuned_math_results.json       # Fine-tuned model results on MATH
├── comparison_report.json            # Complete comparison data
└── comparison_report.html            # Interactive HTML report
```

## Understanding the Output

### Accuracy
- Percentage of correct answers
- Higher is better

### Reasoning Quality Score
- Composite score (0-1) measuring reasoning quality
- Based on:
  - Use of logical keywords (because, therefore, etc.)
  - Step clarity and length
  - Number of reasoning steps

### Reasoning Steps
- Average number of steps in reasoning traces
- More steps may indicate more detailed reasoning

### Sample Outputs
- HTML report includes sample outputs with:
  - Question
  - Step-by-step reasoning
  - Final answer vs ground truth
  - Correctness indicator

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` (default: 4)
- Use `--num-samples` to limit evaluation size
- Use CPU instead of GPU: `--device cpu`

### Missing Datasets
- Script includes sample data for testing
- For full evaluation, place datasets in:
  ```
  data/benchmarks/{benchmark_name}/test.jsonl
  ```

### Slow Evaluation
- Use GPU: `--device cuda`
- Reduce number of samples: `--num-samples 100`
- Evaluate fewer benchmarks: `--benchmarks gsm8k`

## Example Workflow

```bash
# 1. Quick test with small sample
python evaluate.py \
    --base-model gpt2 \
    --benchmarks gsm8k \
    --num-samples 10 \
    --output-dir test_run

# 2. Review results
cat test_run/comparison_report.json | python -m json.tool | head -50

# 3. Full evaluation
python evaluate.py \
    --base-model path/to/base \
    --finetuned-model path/to/finetuned \
    --benchmarks gsm8k math arc mmlu \
    --output-dir full_eval

# 4. Generate visualizations
python visualize_results.py \
    --report full_eval/comparison_report.json \
    --output-dir full_eval/plots

# 5. View results
open full_eval/comparison_report.html
```

## Tips

1. **Start Small**: Use `--num-samples 10` for quick testing
2. **GPU Acceleration**: Use `--device cuda` if available
3. **Reproducibility**: Use `--seed 42` (or any number) for consistent results
4. **Batch Processing**: Use `batch_evaluate.py` for multiple checkpoints
5. **Custom Benchmarks**: Modify dataset classes in `evaluate.py`

## Next Steps

- Read [EVALUATION_README.md](EVALUATION_README.md) for detailed documentation
- Review `eval_config_example.json` for configuration options
- Check `eval_utils.py` for utility functions
- Customize benchmarks for your specific use case
