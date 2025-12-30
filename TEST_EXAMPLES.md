# Test Examples for Evaluation Script

This document provides concrete examples for testing the evaluation system.

## Basic Syntax Check

Verify the script can be run with help:
```bash
python evaluate.py --help
python batch_evaluate.py --help
python visualize_results.py --help
```

## Test with Sample Data

The script includes built-in sample data for testing without external datasets.

### Test 1: Base Model Only (Minimal)
```bash
python evaluate.py \
    --base-model "gpt2" \
    --benchmarks gsm8k \
    --num-samples 3 \
    --output-dir test_output_1
```

**Expected Output:**
- Directory: `test_output_1/`
- Files: `base_gsm8k_results.json`, `comparison_report.json`, `comparison_report.html`
- Terminal: Shows accuracy, reasoning metrics

### Test 2: Multiple Benchmarks
```bash
python evaluate.py \
    --base-model "gpt2" \
    --benchmarks gsm8k math arc \
    --num-samples 3 \
    --output-dir test_output_2 \
    --temperature 0.5
```

**Expected Output:**
- 3 benchmark result files
- Comparison report with 3 benchmarks

### Test 3: Base + Fine-tuned Comparison
```bash
python evaluate.py \
    --base-model "gpt2" \
    --finetuned-model "gpt2" \
    --benchmarks gsm8k \
    --num-samples 3 \
    --output-dir test_output_3
```

**Expected Output:**
- Both base and finetuned result files
- Improvement metrics in comparison report

### Test 4: Custom Configuration
```bash
python evaluate.py \
    --base-model "gpt2" \
    --benchmarks gsm8k math \
    --num-samples 5 \
    --batch-size 2 \
    --temperature 0.7 \
    --device cpu \
    --seed 123 \
    --output-dir test_output_4
```

## Test Batch Evaluation

### Step 1: Create Test Config

Create `test_batch_config.json`:
```json
{
  "global_config": {
    "benchmarks": ["gsm8k"],
    "batch_size": 2,
    "temperature": 0.7,
    "device": "cpu",
    "seed": 42,
    "num_samples": 3
  },
  "evaluations": [
    {
      "name": "test_eval_1",
      "base_model_path": "gpt2",
      "finetuned_model_path": null
    },
    {
      "name": "test_eval_2",
      "base_model_path": "gpt2",
      "finetuned_model_path": "gpt2"
    }
  ]
}
```

### Step 2: Run Batch Evaluation
```bash
python batch_evaluate.py \
    --config test_batch_config.json \
    --output-dir test_batch_output
```

**Expected Output:**
- Directory structure:
  ```
  test_batch_output/
  ├── test_eval_1/
  │   ├── base_gsm8k_results.json
  │   └── comparison_report.json
  ├── test_eval_2/
  │   ├── base_gsm8k_results.json
  │   ├── finetuned_gsm8k_results.json
  │   └── comparison_report.json
  ├── batch_summary.json
  └── comparison_table.txt
  ```

## Test Visualization

```bash
# First run evaluation
python evaluate.py \
    --base-model "gpt2" \
    --finetuned-model "gpt2" \
    --benchmarks gsm8k math \
    --num-samples 3 \
    --output-dir vis_test

# Then generate visualizations
python visualize_results.py \
    --report vis_test/comparison_report.json \
    --output-dir vis_test/plots
```

**Expected Output:**
- `vis_test/plots/accuracy_comparison.png`
- `vis_test/plots/improvement_chart.png`
- `vis_test/plots/reasoning_quality.png`
- `vis_test/plots/reasoning_steps.png`
- `vis_test/plots/radar_gsm8k.png`
- `vis_test/plots/radar_math.png`

## Test Utility Functions

Create `test_utils.py`:
```python
#!/usr/bin/env python3
from eval_utils import (
    normalize_answer,
    extract_numerical_answer,
    extract_multiple_choice_answer,
    calculate_f1_score,
    analyze_reasoning_structure
)

# Test normalization
assert normalize_answer("The answer is 42.") == "the answer is 42"
print("✓ normalize_answer works")

# Test numerical extraction
assert extract_numerical_answer("The answer is 42") == 42.0
assert extract_numerical_answer("= 3.14") == 3.14
print("✓ extract_numerical_answer works")

# Test multiple choice extraction
assert extract_multiple_choice_answer("The answer is A") == "A"
assert extract_multiple_choice_answer("I choose option B") == "B"
print("✓ extract_multiple_choice_answer works")

# Test F1 score
score = calculate_f1_score("the quick brown fox", "quick brown fox")
assert score > 0.8
print("✓ calculate_f1_score works")

# Test reasoning analysis
text = "First, we calculate the sum. Then, we multiply by 2."
analysis = analyze_reasoning_structure(text)
assert analysis['num_lines'] >= 1
assert 'sequential' in analysis['keyword_counts']
print("✓ analyze_reasoning_structure works")

print("\nAll utility tests passed! ✓")
```

Run:
```bash
python test_utils.py
```

## Expected Benchmark Data Structure

If you want to use your own datasets, they should follow these formats:

### GSM8K (data/benchmarks/gsm8k/test.jsonl)
```json
{"question": "Problem text here", "answer": "Solution with answer"}
```

### MATH (data/benchmarks/math/test.jsonl)
```json
{"problem": "Problem text", "solution": "Step by step", "answer": "42"}
```

### ARC (data/benchmarks/arc/test.jsonl)
```json
{"question": "Question text", "choices": ["A", "B", "C", "D"], "answerKey": "A"}
```

### MMLU (data/benchmarks/mmlu/test.jsonl)
```json
{"question": "Question text", "choices": ["A", "B", "C", "D"], "answer": "B"}
```

## Validation Checklist

After running tests, verify:

- [ ] Scripts execute without syntax errors
- [ ] JSON files are valid (use `python -m json.tool file.json`)
- [ ] HTML reports open in browser
- [ ] Accuracy values are between 0 and 1
- [ ] Sample outputs include reasoning steps
- [ ] Improvement percentages are calculated correctly
- [ ] Visualizations are generated (if matplotlib installed)
- [ ] Batch evaluation creates separate directories
- [ ] All metrics are non-negative
- [ ] Terminal output is informative

## Common Issues and Solutions

### Issue: "No module named 'transformers'"
**Solution:** 
```bash
pip install -r requirements-eval.txt
```

### Issue: "CUDA out of memory"
**Solution:**
```bash
python evaluate.py --base-model gpt2 --device cpu ...
```

### Issue: "Dataset not found"
**Solution:** The script uses built-in sample data. No action needed for testing.

### Issue: Visualization fails
**Solution:**
```bash
pip install matplotlib scipy
```

### Issue: Permission denied
**Solution:**
```bash
chmod +x evaluate.py batch_evaluate.py visualize_results.py
```

## Performance Benchmarks

Approximate runtime on CPU with `num_samples=10`:

| Benchmark | Base Only | Base + FT | Full (all benchmarks) |
|-----------|-----------|-----------|----------------------|
| gsm8k     | 2-3 min   | 4-6 min   | -                    |
| math      | 2-3 min   | 4-6 min   | -                    |
| arc       | 1-2 min   | 2-4 min   | -                    |
| mmlu      | 1-2 min   | 2-4 min   | -                    |
| All       | 6-10 min  | 12-20 min | 12-20 min            |

With GPU (CUDA), expect 3-5x speedup.

## Next Steps After Testing

1. **Integrate with your training pipeline:**
   ```bash
   # After training
   python evaluate.py \
       --base-model original_model \
       --finetuned-model ./checkpoints/best_model \
       --benchmarks gsm8k math \
       --output-dir results_$(date +%Y%m%d)
   ```

2. **Set up automated evaluation:**
   - Add to CI/CD pipeline
   - Schedule periodic evaluations
   - Track metrics over time

3. **Customize for your use case:**
   - Add new benchmark datasets
   - Modify reasoning quality metrics
   - Create custom visualizations

4. **Share results:**
   - Email HTML reports
   - Export metrics to tracking systems
   - Create presentation slides from visualizations
