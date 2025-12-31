# Quick Start Guide

Get started with the dataset preparation pipeline in minutes!

## 1. Try the Demo (Recommended)

The easiest way to see the pipeline in action:

```bash
cd data
python demo.py
```

This will process sample datasets and show you exactly how the pipeline works.

## 2. Prepare Your First Dataset

### GSM8K

```bash
python prepare_reasoning_dataset.py \
  --dataset gsm8k \
  --input /path/to/gsm8k_train.jsonl \
  --output ./processed/gsm8k
```

### MATH

```bash
python prepare_reasoning_dataset.py \
  --dataset math \
  --input /path/to/math_train.json \
  --output ./processed/math
```

### ARC

```bash
python prepare_reasoning_dataset.py \
  --dataset arc \
  --input /path/to/ARC-Challenge.jsonl \
  --output ./processed/arc
```

## 3. Check the Output

After running, you'll find three files in your output directory:

- `train.jsonl` - Training set
- `val.jsonl` - Validation set  
- `stats.json` - Dataset statistics

## 4. Customize Settings

### Change train/validation split ratio

```bash
python prepare_reasoning_dataset.py \
  --dataset gsm8k \
  --input input.jsonl \
  --output output/ \
  --train-ratio 0.95  # 95% train, 5% validation
```

### Use JSON format instead of JSONL

```bash
python prepare_reasoning_dataset.py \
  --dataset gsm8k \
  --input input.jsonl \
  --output output/ \
  --format json
```

### Set random seed for reproducibility

```bash
python prepare_reasoning_dataset.py \
  --dataset gsm8k \
  --input input.jsonl \
  --output output/ \
  --seed 12345
```

### Disable shuffling

```bash
python prepare_reasoning_dataset.py \
  --dataset gsm8k \
  --input input.jsonl \
  --output output/ \
  --no-shuffle
```

## 5. Use in Python

```python
from prepare_reasoning_dataset import DatasetPreparator

# Quick one-liner
preparator = DatasetPreparator('gsm8k')
preparator.prepare('input.jsonl', 'output_dir/')

# Or with custom settings
preparator.prepare(
    input_path='input.jsonl',
    output_dir='output_dir/',
    train_ratio=0.95,
    shuffle=True,
    seed=42,
    format='jsonl'
)
```

## Expected Input Formats

### GSM8K Format
```json
{"question": "...", "answer": "reasoning steps\n#### final_answer"}
```

### MATH Format
```json
{"problem": "...", "solution": "...", "answer": "..."}
```

### ARC Format
```json
{
  "question": {"stem": "...", "choices": [...]},
  "answerKey": "A"
}
```

## Output Format

All datasets are converted to the same format:

```json
{
  "question": "Problem statement",
  "reasoning_trace": "Step-by-step solution",
  "answer": "Final answer",
  "metadata": {"dataset": "gsm8k", "source_line": 1}
}
```

## Need Help?

- Read the full documentation: [README.md](README.md)
- Run the demo: `python demo.py`
- See usage examples: `python example_usage.py`
- Check command-line help: `python prepare_reasoning_dataset.py --help`

## Common Issues

### File not found
Make sure the input path is correct and the file exists.

### Empty output
Check if your input file format matches the expected format for the dataset type.

### Validation errors
Use `--strict` flag to see detailed validation errors, or check the logs during processing.
