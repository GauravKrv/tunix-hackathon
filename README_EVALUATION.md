# Model Evaluation System

## ⚠️ IMPORTANT UPDATE

**The evaluation approach has been updated to focus on concrete sample outputs rather than computed quality metrics.**

The previous `evaluate.py` script with subjective reasoning quality scores, coherence metrics, and other derived measures has been **replaced** with a sample-based evaluation approach documented in `samples.md`.

## 📄 New Evaluation Methodology

See **[samples.md](samples.md)** for the current evaluation approach, which:

- Presents concrete before-and-after outputs for identical prompts
- Shows qualitative reasoning differences between model states
- Retains **final answer accuracy** as the only quantitative metric
- Avoids subjective quality ratings that cannot be directly defended
- Focuses on observable differences in reasoning traces

## 🎯 What Changed

### Removed
- ❌ Computed "reasoning quality scores"
- ❌ Coherence metrics
- ❌ Derived quality measures
- ❌ Subjective scoring systems
- ❌ `evaluate.py` script
- ❌ `batch_evaluate.py` script
- ❌ Automated benchmark evaluation with quality scores

### Retained
- ✅ Final answer accuracy (correct/incorrect)
- ✅ Concrete sample outputs
- ✅ Direct comparison of reasoning traces
- ✅ Observable differences in model behavior

## 📊 Current Evaluation Approach

The evaluation now consists of:

1. **Concrete Samples**: Real before-and-after outputs from base and fine-tuned models
2. **Accuracy Metrics**: Simple correct/incorrect classification of final answers
3. **Qualitative Analysis**: Direct observation of reasoning trace differences
4. **No Subjective Scoring**: All comparisons are based on observable output characteristics

## 💡 How to Evaluate Your Model

### 1. Run Inference
Use `inference.py` to generate outputs from your models:

```bash
# Base model
python inference.py \
    --model_path /path/to/base/model \
    --question "Your test question" \
    --output base_output.txt

# Fine-tuned model
python inference.py \
    --model_path /path/to/finetuned/model \
    --question "Your test question" \
    --output finetuned_output.txt
```

### 2. Compare Outputs
Manually compare the outputs to observe:
- Structural differences in reasoning
- Number and organization of steps
- Presence of verification or alternative approaches
- Final answer accuracy

### 3. Document Samples
Add representative samples to your own evaluation document, following the format in `samples.md`.

## 📁 Deprecated Files

The following files are no longer part of the evaluation system:

- `evaluate.py` - Main evaluation script (REMOVED)
- `batch_evaluate.py` - Batch evaluation (depends on evaluate.py, non-functional)
- `run_evaluation.sh` - Example shell script (updated to show deprecation notice)

## 📚 Related Documentation

| Document | Status | Description |
|----------|--------|-------------|
| [samples.md](samples.md) | ✅ **CURRENT** | New evaluation methodology with concrete samples |
| [inference.py](inference.py) | ✅ Active | Use for generating model outputs |
| `eval_utils.py` | ⚠️ Deprecated | Utility functions for old evaluation system |
| `visualize_results.py` | ⚠️ May be deprecated | Visualization for old metrics |

## 🔬 Rationale for Change

The previous evaluation system included computed metrics such as:
- "Reasoning quality scores" based on keyword counting
- "Coherence scores" based on connector presence
- Confidence metrics derived from reasoning characteristics

These metrics were:
1. **Not directly interpretable**: No clear mapping between scores and actual reasoning quality
2. **Not defensible**: Based on heuristics rather than validated criteria
3. **Potentially misleading**: Could suggest quality improvements that aren't real

The new approach focuses on:
1. **Observable differences**: Show actual output, let readers judge
2. **Defensible metrics**: Accuracy is binary and verifiable
3. **Concrete evidence**: Real examples instead of derived scores

## 🛠️ For Advanced Users

If you need quantitative metrics beyond accuracy:

### Option 1: Manual Analysis
Review samples in `samples.md` and count observable features:
- Number of reasoning steps
- Presence of specific keywords or patterns
- Length of responses

### Option 2: Custom Scripts
Write your own evaluation script that:
- Focuses on observable, countable features
- Avoids subjective "quality" aggregations
- Documents exactly what is being measured and why

### Option 3: Human Evaluation
For true quality assessment:
- Have domain experts rate outputs
- Use structured rubrics
- Report inter-rater reliability

## 📞 Support

For questions about the new evaluation approach:
1. Review [samples.md](samples.md) for examples
2. Use `inference.py` to generate your own comparisons
3. Focus on observable differences in outputs

---

**The evaluation system now prioritizes transparency and defensibility over automated scoring.**

See [samples.md](samples.md) to understand the new methodology.
