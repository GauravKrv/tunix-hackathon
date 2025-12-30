# Reasoning Generation Changes

This document describes the implementation of Tasks 1.1 and 1.2, which modify the training pipeline to force the model to generate its own reasoning independently rather than copying gold reasoning traces.

## Overview

The system has been modified to:
1. Remove all `reasoning_trace` fields from the dataset
2. Update all prompt templates with explicit step-by-step reasoning instructions
3. Ensure the model cannot copy gold reasoning structure and must generate reasoning independently
4. Keep the final answer available only for correctness reward calculation

## Task 1.1: Dataset Loader Modifications

### Changes to `data/prepare_reasoning_dataset.py`

1. **ReasoningExample dataclass** - Removed `reasoning_trace` field:
   ```python
   @dataclass
   class ReasoningExample:
       question: str
       answer: str  # Only answer, no reasoning_trace
       metadata: Optional[Dict[str, Any]] = None
   ```

2. **DatasetValidator** - Removed validation of `reasoning_trace`:
   - No longer checks for reasoning_trace existence or content
   - Only validates question and answer fields

3. **GSM8KLoader** - Modified to extract only final answer:
   ```python
   # Extract only the final answer
   if '####' in answer_text:
       parts = answer_text.split('####')
       final_answer = parts[1].strip()
   else:
       final_answer = answer_text.split('\n')[-1].strip()
   ```

4. **MATHLoader** - Removed reasoning trace extraction:
   - Only loads `question` and `answer` fields
   - Ignores `solution` field completely

5. **ARCLoader** - Removed automatic reasoning generation:
   - Only provides question and answer
   - No synthetic reasoning traces created

### Changes to Example Files

Updated `data/example_usage.py` and `data/demo.py`:
- Removed all `reasoning_trace` parameters from ReasoningExample instantiations
- Updated output format documentation to reflect new structure

## Task 1.2: Prompt Template Updates

### New Prompt Template (All Files)

All prompts now include explicit reasoning instructions:

```python
prompt = (
    "You must reason step by step before answering. "
    "Do not give the final answer until reasoning is complete.\n\n"
    f"Question: {question}\n\n"
    "Let's solve this step by step:\n"
)
```

This instruction:
- **Explicitly requires** step-by-step reasoning
- **Prevents premature answers** by instructing to complete reasoning first
- **Replaces implicit structure** from any previously learned patterns

### Files Updated with New Prompt Templates

1. **`train.py`**:
   - `ReasoningDataset._create_prompt()` - New dataset class for loading jsonl files
   - `DummyDataset._create_prompt()` - Updated demo dataset

2. **`inference.py`**:
   - `ReasoningInference.create_prompt()` - Updated default template
   - Other templates (alpaca, chatml, llama2) retain original behavior

3. **`evaluate.py`**:
   - `GSM8KDataset.format_prompt()`
   - `MATHDataset.format_prompt()`
   - `ARCDataset.format_prompt()`
   - `MMLUDataset.format_prompt()`

4. **`example_usage.py`**:
   - `CustomReasoningDataset._create_prompt()`

## New Dataset Classes

### ReasoningDataset (in train.py)

```python
class ReasoningDataset(Dataset):
    """
    Dataset for reasoning tasks where the model generates its own reasoning.
    Only provides prompt (question with explicit reasoning instruction) and final answer.
    No gold reasoning traces are included - model must generate reasoning independently.
    """
```

Features:
- Loads from jsonl files with `question` and `answer` fields only
- Ignores any `reasoning_trace` fields if present
- Creates prompts with explicit reasoning instructions
- Formats as: `prompt + answer` (model generates reasoning between them)

### CustomReasoningDataset (in example_usage.py)

Similar to `ReasoningDataset` but accepts a list of example dicts instead of a file path.

## Verification of Changes

### What the Model Can No Longer Do:
1. ❌ Copy gold reasoning steps from training data (none provided)
2. ❌ Learn implicit reasoning structure from aligned examples (no alignment)
3. ❌ Match step patterns from supervised signals (only final answer supervised)

### What the Model Must Do:
1. ✅ Generate its own reasoning steps independently
2. ✅ Follow explicit instructions to reason step-by-step
3. ✅ Complete reasoning before providing final answer
4. ✅ Use final answer only for correctness reward calculation

## Data Flow

### Training Time:
```
Question → Prompt Template (with explicit reasoning instruction) 
       → Model Generates Reasoning 
       → Model Generates Answer
       → Compare Answer with Ground Truth for Reward
```

### Inference Time:
```
Question → Prompt Template (with explicit reasoning instruction)
       → Model Generates Reasoning 
       → Model Generates Answer
```

## Key Design Decisions

1. **No Reasoning Traces**: Completely removed from data pipeline to prevent any possibility of copying
2. **Explicit Instructions**: Added to all prompt templates to force step-by-step generation
3. **Answer for Reward Only**: Final answer kept solely for computing correctness reward
4. **Consistent Prompts**: Same template used across training, evaluation, and inference

## Files Modified

1. `data/prepare_reasoning_dataset.py` - Dataset loading and preparation
2. `data/example_usage.py` - Example usage with new dataset format
3. `data/demo.py` - Demo scripts with updated examples
4. `train.py` - Added ReasoningDataset, updated DummyDataset
5. `inference.py` - Updated default prompt template
6. `evaluate.py` - Updated all benchmark prompt templates
7. `example_usage.py` - Updated CustomReasoningDataset

## Backward Compatibility

Old datasets with `reasoning_trace` fields will:
- Have the field ignored by loaders (only `question` and `answer` loaded)
- Not cause errors during loading
- Be processed as if only question and answer were provided

## Testing the Changes

To verify the implementation:

1. Prepare a dataset:
   ```bash
   python data/prepare_reasoning_dataset.py --dataset gsm8k --input raw/gsm8k.jsonl --output processed/
   ```

2. Check output format (should only have `question` and `answer`):
   ```bash
   head -n 1 processed/train.jsonl | python -m json.tool
   ```

3. Train with new dataset:
   ```bash
   python train.py --output_dir ./outputs
   ```

4. Verify inference uses correct prompts:
   ```bash
   python inference.py --model_path ./outputs/checkpoint-final --question "What is 2+2?"
   ```

The model should generate reasoning steps independently without access to gold reasoning traces.
