# Model Evaluation System

Complete evaluation framework for testing trained models on reasoning benchmarks with comprehensive metrics, visualizations, and comparison reports.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements-eval.txt

# Run evaluation
python evaluate.py \
    --base-model "gpt2" \
    --benchmarks gsm8k math \
    --num-samples 10 \
    --output-dir results

# View results
open results/comparison_report.html
```

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [File Structure](#file-structure)
- [Examples](#examples)
- [Extending](#extending)

## 🎯 Overview

This evaluation system provides:

- **Multi-Benchmark Support**: GSM8K, MATH, ARC, MMLU
- **Reasoning Analysis**: Extract and score step-by-step reasoning
- **Model Comparison**: Compare base vs fine-tuned models
- **Rich Reports**: JSON, HTML, and visualizations
- **Batch Processing**: Evaluate multiple models at once
- **Built-in Data**: Sample datasets for immediate testing

## ✨ Features

### Evaluation Capabilities
- ✅ Base and fine-tuned model evaluation
- ✅ Multiple reasoning benchmarks
- ✅ Reasoning trace extraction and analysis
- ✅ Quality scoring and metrics
- ✅ GPU/CPU support
- ✅ Reproducible results

### Metrics
- Accuracy (overall and per-benchmark)
- Reasoning quality scores
- Average reasoning steps
- Step length analysis
- Confidence metrics
- Improvement percentages

### Output Formats
- Interactive HTML reports
- JSON data files
- Comparison tables
- Visualization charts
- Sample outputs with traces

## 📦 Installation

```bash
# Basic installation
pip install -r requirements-eval.txt

# With visualization support
pip install -r requirements-eval.txt matplotlib scipy
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA (optional, for GPU acceleration)

## 🔧 Usage

### Basic Evaluation

```bash
python evaluate.py --base-model MODEL_PATH
```

### Compare Models

```bash
python evaluate.py \
    --base-model BASE_MODEL \
    --finetuned-model FINETUNED_MODEL \
    --benchmarks gsm8k math arc mmlu \
    --output-dir results
```

### Batch Evaluation

```bash
python batch_evaluate.py \
    --config batch_config.json \
    --output-dir batch_results
```

### Generate Visualizations

```bash
python visualize_results.py \
    --report results/comparison_report.json \
    --output-dir plots
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [EVALUATION_README.md](EVALUATION_README.md) | Complete documentation and API reference |
| [QUICKSTART_EVALUATION.md](QUICKSTART_EVALUATION.md) | Quick start guide and common workflows |
| [TEST_EXAMPLES.md](TEST_EXAMPLES.md) | Test examples and validation |
| [EVALUATION_CHANGELOG.md](EVALUATION_CHANGELOG.md) | Implementation details and architecture |

## 📁 File Structure

```
├── evaluate.py                    # Main evaluation script
├── eval_utils.py                  # Utility functions
├── batch_evaluate.py              # Batch evaluation
├── visualize_results.py           # Visualization generation
├── run_evaluation.sh              # Example shell script
├── requirements-eval.txt          # Dependencies
├── eval_config_example.json       # Single eval config
├── batch_config_example.json      # Batch eval config
└── Documentation/
    ├── EVALUATION_README.md       # Full documentation
    ├── QUICKSTART_EVALUATION.md   # Quick start
    ├── TEST_EXAMPLES.md           # Test examples
    └── EVALUATION_CHANGELOG.md    # Implementation notes
```

## 💡 Examples

### Example 1: Quick Test
```bash
python evaluate.py \
    --base-model "gpt2" \
    --benchmarks gsm8k \
    --num-samples 5 \
    --output-dir quick_test
```

### Example 2: Full Evaluation
```bash
python evaluate.py \
    --base-model path/to/base/model \
    --finetuned-model path/to/finetuned/model \
    --benchmarks gsm8k math arc mmlu \
    --batch-size 8 \
    --temperature 0.7 \
    --device cuda \
    --output-dir full_evaluation
```

### Example 3: Multiple Checkpoints
```bash
python batch_evaluate.py \
    --config batch_config_example.json \
    --output-dir checkpoint_comparison
```

## 🎨 Sample Output

### Terminal Output
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
```

### Generated Files
```
evaluation_results/
├── base_gsm8k_results.json
├── finetuned_gsm8k_results.json
├── comparison_report.json
└── comparison_report.html
```

## 🔬 Supported Benchmarks

| Benchmark | Type | Description |
|-----------|------|-------------|
| **GSM8K** | Math | Grade school math word problems |
| **MATH** | Math | Advanced mathematics (algebra, calculus, etc.) |
| **ARC** | Science | AI2 Reasoning Challenge |
| **MMLU** | Multi-task | Massive multitask understanding (57 subjects) |

## 🛠️ Extending

### Add Custom Benchmark

```python
class CustomDataset(BenchmarkDataset):
    def load_data(self):
        # Load your data
        pass
    
    def format_prompt(self, item):
        # Format prompt
        pass
    
    def extract_answer(self, text):
        # Extract answer
        pass
    
    def check_answer(self, predicted, ground_truth):
        # Check correctness
        pass
```

### Custom Metrics

Modify `calculate_reasoning_quality()` in `evaluate.py` to implement custom scoring.

### Custom Visualizations

Add plotting functions to `visualize_results.py`.

## 🐛 Troubleshooting

### Out of Memory
```bash
python evaluate.py --device cpu --batch-size 1
```

### Missing Dependencies
```bash
pip install -r requirements-eval.txt
```

### Slow Performance
```bash
python evaluate.py --device cuda --num-samples 100
```

## 📊 Performance

Approximate runtime (num_samples=100):

| Setup | Single Benchmark | All Benchmarks |
|-------|------------------|----------------|
| CPU | 5-10 min | 20-40 min |
| GPU (CUDA) | 1-2 min | 5-10 min |

## 🤝 Contributing

To add features:
1. Review existing code structure
2. Follow established patterns
3. Add tests and documentation
4. Update relevant README files

## 📝 License

See project license file.

## 🙏 Acknowledgments

Built for evaluating reasoning capabilities of language models on standard benchmarks.

## 📞 Support

For issues or questions:
1. Check [TEST_EXAMPLES.md](TEST_EXAMPLES.md) for common issues
2. Review [EVALUATION_README.md](EVALUATION_README.md) for detailed docs
3. Examine example configurations

## 🚦 Status

✅ **Production Ready**
- Fully implemented
- Tested with sample data
- Comprehensive documentation
- Error handling
- Extensible architecture

---

**Start evaluating your models now!**

```bash
python evaluate.py --base-model YOUR_MODEL --num-samples 10
```
