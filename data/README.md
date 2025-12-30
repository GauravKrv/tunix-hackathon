# Dataset Preparation Pipeline

This directory contains the dataset preparation pipeline for reasoning datasets compatible with Tunix.

## Overview

The `prepare_reasoning_dataset.py` script loads, validates, and formats reasoning datasets into a standardized structure with:
- **question**: The problem statement
- **reasoning_trace**: Step-by-step reasoning or solution
- **answer**: The final answer
- **metadata**: Additional dataset-specific information (optional)

## Supported Datasets

### 1. GSM8K (Grade School Math 8K)
- Format: JSONL with `question` and `answer` fields
- Answer format: Reasoning steps followed by `####` separator and final answer
- Example:
  ```json
  {
    "question": "Janet has 3 apples. She buys 2 more. How many does she have?",
    "answer": "Janet starts with 3 apples.\nShe buys 2 more.\n3 + 2 = 5\n#### 5"
  }
  ```

### 2. MATH Dataset
- Format: JSON or JSONL with `problem`/`question`, `solution`, and `answer` fields
- Includes optional metadata: `level`, `type`
- Example:
  ```json
  {
    "problem": "Solve for x: 2x + 3 = 7",
    "solution": "2x + 3 = 7\n2x = 4\nx = 2",
    "answer": "2",
    "level": "Level 1",
    "type": "Algebra"
  }
  ```

### 3. ARC (AI2 Reasoning Challenge)
- Format: JSON or JSONL with `question`, `choices`, and `answerKey` fields
- Converts multiple-choice format to question-reasoning-answer structure
- Example:
  ```json
  {
    "question": {"stem": "What is H2O?", "choices": [...]},
    "answerKey": "A",
    "id": "Mercury_7001"
  }
  ```

## Usage

### Command Line

Basic usage:
```bash
python prepare_reasoning_dataset.py --dataset gsm8k --input path/to/train.jsonl --output output/dir
```

Full options:
```bash
python prepare_reasoning_dataset.py \
  --dataset {gsm8k,math,arc} \
  --input INPUT_FILE \
  --output OUTPUT_DIR \
  --train-ratio 0.9 \
  --seed 42 \
  --format jsonl \
  [--no-shuffle] \
  [--strict]
```

### Arguments

- `--dataset`: Dataset type (required, choices: gsm8k, math, arc)
- `--input`: Path to input dataset file (required)
- `--output`: Output directory for processed dataset (required)
- `--train-ratio`: Ratio of examples for training split (default: 0.9)
- `--no-shuffle`: Disable shuffling before splitting
- `--seed`: Random seed for reproducibility (default: 42)
- `--format`: Output format, jsonl or json (default: jsonl)
- `--strict`: Raise exception on validation errors instead of filtering

### Examples

Prepare GSM8K dataset:
```bash
python prepare_reasoning_dataset.py \
  --dataset gsm8k \
  --input raw/gsm8k_train.jsonl \
  --output processed/gsm8k
```

Prepare MATH dataset with 95/5 split:
```bash
python prepare_reasoning_dataset.py \
  --dataset math \
  --input raw/math_train.json \
  --output processed/math \
  --train-ratio 0.95
```

Prepare ARC dataset in JSON format:
```bash
python prepare_reasoning_dataset.py \
  --dataset arc \
  --input raw/ARC-Challenge.jsonl \
  --output processed/arc \
  --format json
```

## Programmatic Usage

```python
from prepare_reasoning_dataset import DatasetPreparator

# Initialize preparator
preparator = DatasetPreparator('gsm8k')

# Run complete pipeline
preparator.prepare(
    input_path='raw/train.jsonl',
    output_dir='processed/gsm8k',
    train_ratio=0.9,
    shuffle=True,
    seed=42,
    format='jsonl'
)

# Or use individual steps
examples = preparator.load_dataset('raw/train.jsonl')
valid_examples = preparator.validate_dataset(examples)
train, val = preparator.split_dataset(valid_examples, train_ratio=0.9)
preparator.save_dataset(train, 'processed/train.jsonl')
preparator.save_dataset(val, 'processed/val.jsonl')
```

## Output Format

The pipeline generates three files:

### 1. `train.{format}`
Training set in Tunix-compatible format:
```json
{
  "question": "Problem statement",
  "reasoning_trace": "Step-by-step solution",
  "answer": "Final answer",
  "metadata": {
    "dataset": "gsm8k",
    "source_line": 1
  }
}
```

### 2. `val.{format}`
Validation set in the same format as training set.

### 3. `stats.json`
Statistics and metadata about the preparation:
```json
{
  "dataset_type": "gsm8k",
  "total_examples": 7473,
  "valid_examples": 7473,
  "train_examples": 6725,
  "val_examples": 748,
  "train_ratio": 0.9,
  "shuffle": true,
  "seed": 42,
  "format": "jsonl"
}
```

## Data Validation

The pipeline includes comprehensive validation:

- **Non-empty fields**: All required fields must be non-empty strings
- **No whitespace-only content**: Fields cannot contain only whitespace
- **Type checking**: All fields must be of correct type
- **Error reporting**: Invalid examples are logged with detailed error messages

### Validation Modes

- **Default mode** (lenient): Filters out invalid examples and continues
- **Strict mode** (`--strict`): Raises exception on any validation error

## Data Structure

### ReasoningExample Class

```python
@dataclass
class ReasoningExample:
    question: str              # The problem statement
    reasoning_trace: str       # Step-by-step reasoning/solution
    answer: str               # Final answer
    metadata: Optional[Dict]  # Additional dataset-specific info
```

## Extending the Pipeline

To add support for a new dataset:

1. Create a new loader class:
```python
class NewDatasetLoader:
    @staticmethod
    def load(file_path: str) -> List[ReasoningExample]:
        # Implement loading logic
        pass
```

2. Register the loader:
```python
DatasetPreparator.LOADERS['newdataset'] = NewDatasetLoader
```

## Requirements

- Python 3.7+
- Standard library only (no external dependencies)

## Directory Structure

```
data/
├── README.md                      # This file
├── prepare_reasoning_dataset.py   # Main pipeline script
├── raw/                          # Raw dataset files (not included)
│   ├── gsm8k_train.jsonl
│   ├── math_train.json
│   └── arc_challenge.jsonl
└── processed/                    # Processed datasets (generated)
    ├── gsm8k/
    │   ├── train.jsonl
    │   ├── val.jsonl
    │   └── stats.json
    ├── math/
    │   ├── train.jsonl
    │   ├── val.jsonl
    │   └── stats.json
    └── arc/
        ├── train.jsonl
        ├── val.jsonl
        └── stats.json
```
