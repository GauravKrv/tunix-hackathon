# Test Examples for Evaluation - DEPRECATED

## ⚠️ Evaluation System Has Changed

The automated evaluation scripts referenced in this document have been **deprecated**. This document is retained for historical reference only.

## What Changed

The `evaluate.py` and `batch_evaluate.py` scripts have been removed and replaced with a sample-based evaluation approach. See [samples.md](samples.md) for the current methodology.

## New Testing Approach

### Test 1: Generate Sample Output
```bash
python inference.py \
    --model_path "path/to/model" \
    --question "A store has 20 apples. They sell 8 apples in the morning and 5 in the afternoon. How many apples are left?"
```

**Expected Output:**
- Model's step-by-step reasoning
- Final answer

### Test 2: Compare Base vs Fine-tuned
```bash
# Base model
python inference.py \
    --model_path "path/to/base/model" \
    --question "John has 3 times as many marbles as Jane. If Jane has 12 marbles, how many does John have?"

# Fine-tuned model
python inference.py \
    --model_path "path/to/finetuned/model" \
    --question "John has 3 times as many marbles as Jane. If Jane has 12 marbles, how many does John have?"
```

**Expected Output:**
- Two outputs to manually compare
- Observe differences in reasoning structure
- Check final answer accuracy

### Test 3: Multiple Questions
Create a test set and run inference on each:
```bash
# Create test_questions.txt with one question per line
# Then for each question:
while read question; do
    echo "Base: $question"
    python inference.py --model_path BASE_MODEL --question "$question"
    echo "Fine-tuned: $question"
    python inference.py --model_path FT_MODEL --question "$question"
    echo "---"
done < test_questions.txt
```

## Validation Checklist

Instead of automated metrics, validate by:

- [ ] Both models generate valid outputs
- [ ] Outputs contain reasoning steps
- [ ] Final answers are extractable
- [ ] Answers can be compared to ground truth
- [ ] Differences between models are observable

## Expected Observations

Rather than numeric scores, look for:

- **Structure**: Is reasoning more organized in one model?
- **Steps**: Does one model use more reasoning steps?
- **Clarity**: Is one model's explanation clearer?
- **Accuracy**: Does one model answer correctly more often?

## Common Issues

### Issue: Model not loading
```bash
# Check model path
ls path/to/model
# Should contain model files (pytorch_model.bin, config.json, etc.)
```

### Issue: Out of memory
```bash
# Use CPU or smaller model
python inference.py --model_path MODEL --question "..." --device cpu
```

### Issue: Slow inference
```bash
# This is normal for large models
# Consider using GPU if available
```

## Reference

For the current evaluation methodology, see:
- [samples.md](samples.md) - Concrete evaluation examples
- [README_EVALUATION.md](README_EVALUATION.md) - Overview of changes
- [QUICKSTART_EVALUATION.md](QUICKSTART_EVALUATION.md) - Updated guide

---

**This document describes the old evaluation system and is retained for historical reference only.**
