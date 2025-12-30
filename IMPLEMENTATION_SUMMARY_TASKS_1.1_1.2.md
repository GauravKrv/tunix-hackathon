# Implementation Summary: Tasks 1.1 and 1.2

## Objective
Modify the training pipeline to force the model to generate its own reasoning independently, without copying gold reasoning traces from the training data.

## Task 1.1: Remove Reasoning Traces from Dataset Loader

### Changes Made

#### 1. `data/prepare_reasoning_dataset.py`

**Modified `ReasoningExample` dataclass:**
- **Removed**: `reasoning_trace: str` field
- **Kept**: `question: str`, `answer: str`, `metadata: Optional[Dict]`
- **Result**: Dataset now only contains question and final answer

**Updated `DatasetValidator.validate_example()`:**
- **Removed**: All validation logic for `reasoning_trace`
- **Kept**: Validation for `question` and `answer` fields only
- **Result**: No longer checks for reasoning trace presence or validity

**Modified `GSM8KLoader.load()`:**
- **Changed**: Extracts only final answer from `####` separator
- **Removed**: Extraction of reasoning steps before `####`
- **Result**: Gold reasoning steps are completely discarded

**Modified `MATHLoader._parse_math_example()`:**
- **Removed**: Loading of `solution` field containing reasoning
- **Kept**: Only `question` and `answer` fields
- **Result**: Solution/reasoning field is ignored entirely

**Modified `ARCLoader._parse_arc_example()`:**
- **Removed**: Automatic generation of reasoning traces
- **Removed**: Logic to create synthetic reasoning from facts
- **Result**: No reasoning provided for multiple choice questions

**Updated module docstring:**
- Changed from "question-reasoning_trace-answer structure"
- To: "question-answer structure" with note that model generates reasoning

#### 2. `data/example_usage.py`

**Updated example data:**
- Removed all `reasoning_trace` parameters from `ReasoningExample` instantiations
- Updated 3 separate example functions with new format

#### 3. `data/demo.py`

**Updated demo examples:**
- Removed `reasoning_trace` from all custom examples
- Updated output format explanation to reflect new structure
- Added note: "Model generates its own reasoning during training"

## Task 1.2: Update Prompt Templates with Explicit Reasoning Instructions

### New Prompt Template

All prompt templates now include explicit instruction:

```python
prompt = (
    "You must reason step by step before answering. "
    "Do not give the final answer until reasoning is complete.\n\n"
    f"Question: {question}\n\n"
    "Let's solve this step by step:\n"
)
```

### Changes Made

#### 1. `train.py`

**Added new `ReasoningDataset` class:**
- Loads data from jsonl files with only `question` and `answer`
- Implements `_create_prompt()` with explicit reasoning instruction
- Ignores any `reasoning_trace` fields if present in old data
- Formats training examples as: `prompt + answer` (model fills in reasoning)

**Updated `DummyDataset` class:**
- Changed from generic text prompts to question-answer format
- Added `_create_prompt()` method with explicit reasoning instruction
- Updated example questions to reasoning tasks
- Maintains compatibility for demo/testing purposes

#### 2. `inference.py`

**Modified `ReasoningInference.create_prompt()` method:**
- Updated `default` template to match training prompt exactly
- Added explicit reasoning instructions
- Maintains other templates (alpaca, chatml, llama2) for compatibility
- Ensures inference uses same prompt structure as training

#### 3. `evaluate.py`

**Updated 4 benchmark dataset classes:**

**`GSM8KDataset.format_prompt()`:**
- Replaced "Solve the following math problem. Show your reasoning step by step."
- With: Explicit reasoning instruction template

**`MATHDataset.format_prompt()`:**
- Replaced "Solve the following math problem. Provide detailed reasoning."
- With: Explicit reasoning instruction template

**`ARCDataset.format_prompt()`:**
- Replaced "Answer the following question... Explain your reasoning."
- With: Explicit reasoning instruction template

**`MMLUDataset.format_prompt()`:**
- Replaced "Answer the following... Provide reasoning for your answer."
- With: Explicit reasoning instruction template

#### 4. `example_usage.py`

**Created new `CustomReasoningDataset` class:**
- Replaced old `CustomTextDataset`
- Implements explicit reasoning prompt template
- Works with question-answer format only
- Added `_create_prompt()` method

**Updated helper functions:**
- `create_sample_dataset()`: Changed to return reasoning task examples
- `_mp_fn_custom()`: Updated to use `CustomReasoningDataset`
- `run_example_1()` and `run_example_2()`: Updated variable names

## Verification of Implementation

### Task 1.1 Verification ✅

**Reasoning traces completely removed:**
- ✅ `ReasoningExample` dataclass no longer has `reasoning_trace` field
- ✅ No validation of reasoning traces
- ✅ GSM8K loader discards all text before `####`
- ✅ MATH loader ignores `solution` field
- ✅ ARC loader provides no synthetic reasoning
- ✅ All example files updated to not include reasoning traces

**Only prompt and final answer retained:**
- ✅ Dataset loaders extract only `question` and `answer`
- ✅ Prompt is created on-the-fly with explicit instructions
- ✅ Final answer is kept for correctness reward calculation

### Task 1.2 Verification ✅

**All prompts include explicit reasoning instruction:**
- ✅ Training dataset classes (`ReasoningDataset`, `DummyDataset`)
- ✅ Inference default template
- ✅ All 4 evaluation benchmark templates (GSM8K, MATH, ARC, MMLU)
- ✅ Custom example dataset

**Instruction content verified:**
- ✅ States: "You must reason step by step before answering"
- ✅ States: "Do not give the final answer until reasoning is complete"
- ✅ Consistent across all uses

### End-to-End Verification ✅

**Model cannot copy gold reasoning:**
- ✅ No reasoning traces in data → nothing to copy
- ✅ No alignment between generated and gold steps → no supervision on steps
- ✅ Only final answer available → used solely for reward

**Model must generate reasoning independently:**
- ✅ Explicit instruction forces step-by-step generation
- ✅ Prompt structure encourages reasoning before answer
- ✅ Same template across training/eval/inference

## Files Modified

1. **`data/prepare_reasoning_dataset.py`** - Core dataset preparation
2. **`data/example_usage.py`** - Example dataset usage
3. **`data/demo.py`** - Demo scripts
4. **`train.py`** - Training datasets and prompts
5. **`inference.py`** - Inference prompts
6. **`evaluate.py`** - Evaluation benchmark prompts
7. **`example_usage.py`** - Custom training datasets

## Additional Documentation

Created **`REASONING_CHANGES.md`** with:
- Detailed explanation of all changes
- Rationale for design decisions
- Testing instructions
- Backward compatibility notes

## Summary

Both tasks have been **fully implemented**:

1. ✅ **Task 1.1**: All reasoning traces removed from dataset pipeline
   - Dataset loader modifications complete
   - Alignment logic completely removed
   - Only question and answer retained

2. ✅ **Task 1.2**: All prompt templates updated with explicit instructions
   - Consistent reasoning instruction across all files
   - Replaces implicit structure from training data
   - Forces independent reasoning generation

The model now:
- **Cannot** copy gold reasoning (none available)
- **Cannot** learn from aligned supervision (no alignment)
- **Must** generate its own reasoning independently
- **Uses** final answer only for correctness reward calculation
