# Model Evaluation - DEPRECATED

## ⚠️ THIS SCRIPT HAS BEEN DEPRECATED

The `evaluate.py` script and associated automated evaluation system have been **removed** and replaced with a sample-based evaluation approach.

## Why Was This Deprecated?

The previous evaluation system computed metrics such as:
- "Reasoning quality scores" based on keyword counting
- "Coherence scores" based on connector word presence  
- Confidence metrics derived from reasoning characteristics

These metrics were:
1. **Not directly interpretable**: Unclear what a score of 0.75 means for actual reasoning quality
2. **Not defensible**: Based on heuristics without validation
3. **Potentially misleading**: Could suggest improvements that don't reflect true capability changes

## New Evaluation Approach

See **[samples.md](samples.md)** for the current evaluation methodology.

The new approach:
- ✅ Presents concrete before-and-after outputs for identical prompts
- ✅ Shows qualitative reasoning differences between model states
- ✅ Retains **final answer accuracy** as the only quantitative metric
- ✅ Avoids subjective quality ratings that cannot be directly defended
- ✅ Focuses on observable differences in reasoning traces

## What to Use Instead

### For Generating Outputs
Use `inference.py` to generate model outputs:

```bash
# Base model
python inference.py \
    --model_path /path/to/base/model \
    --question "Your test question"

# Fine-tuned model  
python inference.py \
    --model_path /path/to/finetuned/model \
    --question "Your test question"
```

### For Evaluation
1. Generate outputs from both models on identical prompts
2. Manually compare the outputs
3. Calculate accuracy (correct/incorrect) for quantitative metric
4. Document qualitative differences in reasoning structure
5. See [samples.md](samples.md) for examples

### For Documentation
Follow the format in [samples.md](samples.md):
- Show the prompt
- Show both model outputs
- Note the final answer and correctness
- Describe observable differences

## Deprecated Files

The following files are no longer functional:
- `evaluate.py` - REMOVED
- `batch_evaluate.py` - Non-functional (depends on evaluate.py)
- Functions in `eval_utils.py` that compute quality/coherence scores

## Migration Guide

If you were using the old evaluation system:

### Before
```bash
python evaluate.py \
    --base-model BASE_MODEL \
    --finetuned-model FINETUNED_MODEL \
    --benchmarks gsm8k math \
    --output-dir results
```

### After
```bash
# Generate samples from base model
python inference.py --model_path BASE_MODEL --question "QUESTION_1" > base_q1.txt
python inference.py --model_path BASE_MODEL --question "QUESTION_2" > base_q2.txt
# ... for all test questions

# Generate samples from fine-tuned model
python inference.py --model_path FINETUNED_MODEL --question "QUESTION_1" > ft_q1.txt
python inference.py --model_path FINETUNED_MODEL --question "QUESTION_2" > ft_q2.txt
# ... for all test questions

# Manually compare outputs and document in your own samples file
```

## Reference Documentation

- [samples.md](samples.md) - **Current evaluation methodology**
- [README_EVALUATION.md](README_EVALUATION.md) - Overview of changes
- [QUICKSTART_EVALUATION.md](QUICKSTART_EVALUATION.md) - Updated quick start guide

## For Advanced Users

If you need quantitative metrics beyond accuracy:

### Option 1: Count Observable Features
Write scripts that count specific, observable features:
- Number of reasoning steps (by counting lines/sentences)
- Presence of specific keywords (count, don't score)
- Length of outputs (characters/words)
- Structural patterns (numbered lists, etc.)

### Option 2: Human Evaluation
For true quality assessment:
- Have domain experts rate outputs
- Use structured rubrics with clear criteria
- Report inter-rater reliability
- Document the rating process

### Option 3: Established Benchmarks
Use benchmarks with ground truth answers:
- Report accuracy only
- Don't derive quality scores from reasoning characteristics
- Let accuracy speak for itself

## Contact

For questions about the new evaluation approach:
1. Review [samples.md](samples.md) for concrete examples
2. Use `inference.py` to generate your own comparisons
3. Focus on observable, verifiable differences

---

**The evaluation system now prioritizes transparency and defensibility over automated scoring.**
