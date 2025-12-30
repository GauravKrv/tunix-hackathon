# Model Evaluation Script

Comprehensive evaluation script for testing trained models on reasoning benchmarks.

## Features

- **Multi-benchmark Support**: Evaluate on GSM8K, MATH, ARC, and MMLU benchmarks
- **Reasoning Trace Analysis**: Extract and analyze step-by-step reasoning from model outputs
- **Quality Metrics**: Compute accuracy, reasoning quality scores, and confidence metrics
- **Model Comparison**: Generate detailed comparison reports between base and fine-tuned models
- **Rich Output Formats**: JSON and HTML reports with sample outputs and visualizations

## Installation

```bash
pip install -r requirements-eval.txt
```

## Usage

### Basic Usage

Evaluate a base model only:

```bash
python evaluate.py --base-model /path/to/base/model
```

### Compare Base and Fine-tuned Models

```bash
python evaluate.py \
    --base-model /path/to/base/model \
    --finetuned-model /path/to/finetuned/model \
    --benchmarks gsm8k math arc mmlu \
    --output-dir evaluation_results
```

### Custom Evaluation Settings

```bash
python evaluate.py \
    --base-model /path/to/base/model \
    --finetuned-model /path/to/finetuned/model \
    --benchmarks gsm8k math \
    --num-samples 100 \
    --batch-size 8 \
    --temperature 0.7 \
    --device cuda \
    --output-dir my_evaluation_results
```

## Command-line Arguments

- `--base-model`: Path to the base model (required)
- `--finetuned-model`: Path to the fine-tuned model (optional)
- `--benchmarks`: List of benchmarks to evaluate on (default: gsm8k math arc mmlu)
- `--output-dir`: Directory for saving results (default: evaluation_results)
- `--batch-size`: Batch size for evaluation (default: 4)
- `--num-samples`: Number of samples to evaluate per benchmark (default: all)
- `--temperature`: Sampling temperature for generation (default: 0.7)
- `--device`: Device to use (default: cuda if available, else cpu)
- `--seed`: Random seed for reproducibility (default: 42)

## Supported Benchmarks

### GSM8K
Grade school math word problems requiring multi-step reasoning.

### MATH
Advanced mathematics problems covering algebra, geometry, calculus, and more.

### ARC (AI2 Reasoning Challenge)
Science questions requiring reasoning and world knowledge.

### MMLU (Massive Multitask Language Understanding)
Multiple-choice questions across 57 subjects.

## Output

The script generates the following outputs in the specified output directory:

### JSON Files
- `base_{benchmark}_results.json`: Detailed results for base model on each benchmark
- `finetuned_{benchmark}_results.json`: Detailed results for fine-tuned model on each benchmark
- `comparison_report.json`: Complete comparison data in JSON format

### HTML Report
- `comparison_report.html`: Interactive HTML report with:
  - Summary table of improvements across benchmarks
  - Detailed results for each benchmark
  - Sample outputs with reasoning traces
  - Visual highlighting of correct/incorrect predictions

## Metrics

The script computes the following metrics:

- **Accuracy**: Percentage of correct predictions
- **Reasoning Steps**: Average number of reasoning steps per problem
- **Reasoning Length**: Average word count per reasoning step
- **Reasoning Quality Score**: Composite score based on:
  - Presence of reasoning keywords (because, therefore, etc.)
  - Step length and clarity
  - Number of logical steps

## Dataset Format

The script expects benchmark datasets in the following structure:

```
data/
└── benchmarks/
    ├── gsm8k/
    │   └── test.jsonl
    ├── math/
    │   └── test.jsonl
    ├── arc/
    │   └── test.jsonl
    └── mmlu/
        └── test.jsonl
```

If datasets are not found, the script will use built-in sample data for testing.

### GSM8K Format
```json
{"question": "A store has 20 apples...", "answer": "The answer is 7."}
```

### MATH Format
```json
{"problem": "Solve for x: 2x + 5 = 13", "solution": "...", "answer": "4"}
```

### ARC Format
```json
{"question": "Which property...", "choices": ["weight", "color", ...], "answerKey": "A"}
```

### MMLU Format
```json
{"question": "What is...", "choices": ["Option A", "Option B", ...], "answer": "B"}
```

## Example Output

```
================================================================================
EVALUATION SUMMARY
================================================================================

gsm8k:
  Base Accuracy: 45.20%
  Fine-tuned Accuracy: 67.80%
  Improvement: +22.60% (+50.0%)
  Quality Score Change: +0.152

math:
  Base Accuracy: 23.50%
  Fine-tuned Accuracy: 38.90%
  Improvement: +15.40% (+65.5%)
  Quality Score Change: +0.203

Detailed results saved to: evaluation_results
View HTML report at: evaluation_results/comparison_report.html
```

## Customization

### Adding New Benchmarks

To add a new benchmark, create a class inheriting from `BenchmarkDataset`:

```python
class MyBenchmarkDataset(BenchmarkDataset):
    def load_data(self) -> List[Dict]:
        # Load your dataset
        pass
    
    def format_prompt(self, item: Dict) -> str:
        # Format prompt for model
        pass
    
    def extract_answer(self, text: str) -> str:
        # Extract answer from model output
        pass
    
    def check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Check if answer is correct
        pass
```

Then register it in the `dataset_map` dictionary in `ModelEvaluator.run_evaluation()`.

### Custom Reasoning Quality Metrics

Modify the `calculate_reasoning_quality()` method in `ModelEvaluator` to implement custom quality metrics.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA (optional, for GPU acceleration)

## Notes

- GPU is highly recommended for efficient evaluation
- Evaluation time depends on model size, number of benchmarks, and samples
- The script automatically handles model loading, device placement, and memory management
- All outputs include timestamps and are reproducible with the seed parameter
