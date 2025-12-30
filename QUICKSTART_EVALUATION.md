# Quick Start Guide for Model Evaluation

## ⚠️ IMPORTANT: Evaluation System Updated

**The automated evaluation scripts have been deprecated.** This guide has been updated to reflect the new sample-based evaluation approach.

## New Evaluation Approach

The evaluation system now focuses on **concrete sample outputs** rather than computed quality metrics. See [samples.md](samples.md) for the complete methodology and examples.

## Quick Start: Generate Sample Outputs

### 1. Run Inference on Test Questions

Generate outputs from your base model:
```bash
python inference.py \
    --model_path path/to/base/model \
    --question "A store has 20 apples. They sell 8 apples in the morning and 5 in the afternoon. How many apples are left?"
```

Generate outputs from your fine-tuned model:
```bash
python inference.py \
    --model_path path/to/finetuned/model \
    --question "A store has 20 apples. They sell 8 apples in the morning and 5 in the afternoon. How many apples are left?"
```

### 2. Compare Outputs

Manually compare the outputs to observe:
- **Structure**: How is the reasoning organized?
- **Steps**: How many reasoning steps are present?
- **Verification**: Does the model verify its answer?
- **Accuracy**: Is the final answer correct?

### 3. Document Your Findings

Create your own evaluation document following the format in [samples.md](samples.md):

```markdown
## Sample: Your Test Case

### Prompt
[Your question]

### Base Model Output
[Output from base model]

**Final Answer**: [extracted answer]
**Ground Truth**: [correct answer]
**Accuracy**: [✓ or ✗]

### Fine-tuned Model Output
[Output from fine-tuned model]

**Final Answer**: [extracted answer]
**Ground Truth**: [correct answer]
**Accuracy**: [✓ or ✗]

### Reasoning Trace Differences
[Your observations about how the outputs differ]
```

## What Changed

### ❌ Removed
- Automated `evaluate.py` script
- Reasoning quality scores
- Coherence metrics
- Batch evaluation with derived metrics
- HTML reports with subjective scores

### ✅ Current Approach
- Manual sample generation using `inference.py`
- Direct observation of output differences
- Final answer accuracy only
- Concrete examples in `samples.md`

## Why This Change?

The previous evaluation system included metrics like "reasoning quality scores" that were:
1. Based on heuristics (keyword counting, etc.)
2. Not directly interpretable
3. Not defensible without human validation

The new approach:
1. Shows actual outputs for direct comparison
2. Measures only accuracy (binary, verifiable)
3. Lets users observe differences without subjective scoring

## Workflow for Your Evaluation

### Step 1: Select Test Questions
Choose representative questions from your domain:
- Math problems (if training on math)
- Science questions (if training on reasoning)
- Domain-specific problems (for specialized training)

### Step 2: Generate Outputs
Run inference on both models for each question:
```bash
# For each test question
python inference.py --model_path BASE_MODEL --question "QUESTION"
python inference.py --model_path FINETUNED_MODEL --question "QUESTION"
```

### Step 3: Analyze Results
For each sample, document:
- The reasoning steps in each output
- The final answer from each model
- Whether each answer is correct
- Observable differences in reasoning style

### Step 4: Calculate Accuracy
Count correct answers:
- Base model: X/N correct
- Fine-tuned model: Y/N correct

This is your quantitative metric.

### Step 5: Document Qualitative Observations
Note patterns in reasoning differences:
- Does fine-tuned model use more structured steps?
- Does it include verification?
- Does it make errors in different places?
- Does it provide more or less explanation?

## Example Usage

```bash
# Test question 1
python inference.py \
    --model_path ./base_model \
    --question "John has 3 times as many marbles as Jane. If Jane has 12 marbles, how many does John have?"

python inference.py \
    --model_path ./finetuned_model \
    --question "John has 3 times as many marbles as Jane. If Jane has 12 marbles, how many does John have?"

# Test question 2
python inference.py \
    --model_path ./base_model \
    --question "What is the value of x in the equation 2x + 5 = 13?"

python inference.py \
    --model_path ./finetuned_model \
    --question "What is the value of x in the equation 2x + 5 = 13?"

# Continue for all test questions...
```

## Tips for Effective Evaluation

1. **Use Identical Prompts**: Ensure both models receive exactly the same prompt
2. **Test Diverse Cases**: Include easy, medium, and hard problems
3. **Check Edge Cases**: Test unusual or tricky questions
4. **Document Everything**: Keep all outputs for reference
5. **Focus on Patterns**: Look for consistent differences across samples
6. **Be Objective**: Describe what you observe, not what you hope to see

## Understanding Results

### Accuracy Metrics
- Simple ratio: correct answers / total questions
- Report separately for base and fine-tuned models
- This is your primary quantitative metric

### Qualitative Observations
Instead of scores, note observable features:
- "Fine-tuned model consistently numbers its steps"
- "Base model provides shorter explanations"
- "Fine-tuned model includes verification in 7/10 samples"
- "Both models struggle with multi-step word problems"

## Reference Documentation

- [samples.md](samples.md) - Full evaluation methodology with examples
- [README_EVALUATION.md](README_EVALUATION.md) - Overview of the change
- [inference.py](inference.py) - Script for generating outputs

## Questions?

The new evaluation approach prioritizes:
- **Transparency**: Show actual outputs
- **Defensibility**: Measure only what can be verified
- **Simplicity**: Avoid complex derived metrics

If you need automated metrics beyond accuracy, consider:
1. Writing custom scripts for specific, observable features
2. Having human experts rate outputs
3. Using established benchmarks with ground truth

---

**Start evaluating with concrete samples now:**
```bash
python inference.py --model_path YOUR_MODEL --question "YOUR_QUESTION"
```
