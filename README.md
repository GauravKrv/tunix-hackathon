# Tunix/JAX TPU Training Project

A scalable machine learning training framework built with Tunix, JAX, and optimized for TPU acceleration.

## Project Overview

This project provides a flexible infrastructure for training machine learning models using JAX on TPU hardware, with support for custom reward functions, distributed training, and comprehensive evaluation pipelines.

---

## Inference Script for Fine-Tuned Reasoning Models

This repository includes an inference script for loading fine-tuned language models and generating step-by-step reasoning for custom questions.

### Inference Features

- Load fine-tuned models from local paths or HuggingFace Hub
- Multiple prompt template formats (default, Alpaca, ChatML, Llama2)
- Customizable generation parameters
- Clear step-by-step reasoning display
- Batch processing from JSON files
- Support for 8-bit and 4-bit quantization
- Formatted output with reasoning step parsing

### Quick Inference Usage

Run inference with a single question:

```bash
python inference.py \
    --model_path /path/to/fine-tuned-model \
    --question "If a train travels 120 miles in 2 hours, what is its average speed?"
```

Process multiple questions from a JSON file:

```bash
python inference.py \
    --model_path /path/to/fine-tuned-model \
    --questions_file example_questions.json \
    --output_file results.json
```

See [Inference Documentation](#inference-documentation) below for complete details.

---

## Dataset Preparation Pipeline

A comprehensive, production-ready pipeline for preparing reasoning datasets (GSM8K, MATH, ARC) in Tunix-compatible format.

### 🚀 Quick Start

```bash
# Run the demo with sample data
cd data
python demo.py

# Or process your own dataset
python prepare_reasoning_dataset.py \
  --dataset gsm8k \
  --input /path/to/gsm8k_train.jsonl \
  --output processed/gsm8k
```

### 📦 What's Included

- **Complete Implementation**: 576-line pipeline supporting GSM8K, MATH, and ARC datasets
- **Zero Dependencies**: Uses only Python standard library (3.7+)
- **Comprehensive Documentation**: Full API docs, quick start guide, and examples
- **Sample Data**: Test datasets for all three formats
- **Interactive Demos**: See the pipeline in action
- **Multiple Interfaces**: CLI, Python API, and configuration-based usage

### ✨ Features

#### Supported Datasets
- **GSM8K**: Grade School Math word problems
- **MATH**: Competition-level mathematics problems
- **ARC**: AI2 Reasoning Challenge (science questions)

#### Core Capabilities
- Load datasets from JSON or JSONL files
- Validate data quality and integrity
- Split into train/validation sets (configurable ratios)
- Save in Tunix-compatible format
- Generate statistics and metadata
- Preserve original metadata
- Handle errors gracefully

#### Output Format
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

### 📖 Documentation

- **[data/QUICKSTART.md](data/QUICKSTART.md)** - Get started in 5 minutes
- **[data/README.md](data/README.md)** - Complete API documentation
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Architecture overview
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Feature summary

---

## Gemma2 2B Training with Tunix Trainer

This repository now includes a complete training pipeline for fine-tuning Google's Gemma2 2B model using a custom Tunix trainer optimized for TPU infrastructure.

### Features

- **Gemma2 2B Model**: Pre-trained causal language model from Google
- **TPU Optimization**: Full support for TPU v2/v3/v4 with XLA compilation
- **Composite Reward Functions**: Multi-objective optimization with:
  - Quality reward (output-target alignment)
  - Safety reward (content safety via entropy)
  - Diversity reward (output variety)
  - Coherence reward (representation consistency)
- **Training Features**:
  - Gradient accumulation
  - Gradient checkpointing
  - Mixed precision training (bfloat16)
  - Learning rate scheduling with warmup
  - Distributed training across TPU cores
  - Automatic checkpointing with rotation
  - Comprehensive logging

---

## Features

- **TPU Acceleration**: Optimized for Google Cloud TPUs with JAX
- **Modular Architecture**: Separate components for configs, datasets, reward functions, training, and evaluation
- **Distributed Training**: Built-in support for multi-device and multi-host training
- **Custom Reward Functions**: Flexible reward function framework for reinforcement learning and custom objectives
- **Reproducible Experiments**: Configuration-driven approach for experiment management

## Project Structure

```
.
├── configs/              # Configuration files for experiments
├── datasets/            # Dataset loading and preprocessing
├── reward_functions/    # Custom reward function implementations
├── training/            # Training scripts and utilities
├── evaluation/          # Evaluation metrics and scripts
├── data/                # Dataset preparation pipeline
│   ├── prepare_reasoning_dataset.py   # Main pipeline (576 lines)
│   ├── README.md                      # Comprehensive documentation
│   ├── QUICKSTART.md                  # 5-minute getting started guide
│   ├── demo.py                        # Interactive demonstrations
│   ├── example_usage.py               # Usage examples
│   ├── config_example.py              # Configuration templates
│   └── sample_data/                   # Sample datasets
├── train.py             # Main Gemma2 training script
├── utils.py             # Training utilities
├── example_usage.py     # Training examples
├── inference.py         # Inference script for fine-tuned models
├── example_questions.json  # Sample questions for inference
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## Installation

### Prerequisites

- Python 3.9 or higher (3.7+ for dataset pipeline only)
- Access to TPU hardware (Google Cloud TPU or Colab TPU)
- CUDA-compatible GPU (optional, for local development)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For TPU setup on Google Cloud:
```bash
# Install TPU-specific dependencies
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Quick Start

### Gemma2 Training

#### Basic Training

```bash
python train.py \
  --model_name google/gemma-2-2b \
  --output_dir ./outputs \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5
```

#### Advanced Configuration

```bash
python train.py \
  --model_name google/gemma-2-2b \
  --output_dir ./outputs \
  --num_epochs 5 \
  --batch_size 8 \
  --learning_rate 3e-5 \
  --warmup_steps 200 \
  --max_length 2048 \
  --gradient_accumulation_steps 2 \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 100 \
  --num_tpu_cores 8
```

### Dataset Preparation

```bash
# Process a reasoning dataset
python data/prepare_reasoning_dataset.py \
  --dataset gsm8k \
  --input raw/train.jsonl \
  --output processed/gsm8k
```

## Inference Documentation

### Basic Usage

Run inference with a single question:

```bash
python inference.py \
    --model_path /path/to/fine-tuned-model \
    --question "If a train travels 120 miles in 2 hours, what is its average speed?"
```

### Run with Example Questions

If no question is provided, the script will run with default examples:

```bash
python inference.py --model_path /path/to/fine-tuned-model
```

### Batch Processing

Process multiple questions from a JSON file:

```bash
python inference.py \
    --model_path /path/to/fine-tuned-model \
    --questions_file questions.json \
    --output_file results.json
```

Example `questions.json` format:

```json
[
    "What is 15% of 200?",
    "A car travels 60 km/h for 3 hours. How far does it go?",
    "Solve: 3x - 7 = 14"
]
```

Or:

```json
{
    "questions": [
        "What is 15% of 200?",
        "A car travels 60 km/h for 3 hours. How far does it go?"
    ]
}
```

### Prompt Templates

Choose different prompt formats based on your model's training:

```bash
# Alpaca format
python inference.py \
    --model_path /path/to/model \
    --question "Your question here" \
    --prompt_template alpaca

# ChatML format
python inference.py \
    --model_path /path/to/model \
    --question "Your question here" \
    --prompt_template chatml

# Llama2 format
python inference.py \
    --model_path /path/to/model \
    --question "Your question here" \
    --prompt_template llama2
```

### Custom System Message

Provide a custom system message:

```bash
python inference.py \
    --model_path /path/to/model \
    --question "Your question here" \
    --system_message "You are a math tutor. Explain each step clearly for a student."
```

### Generation Parameters

Control the generation behavior:

```bash
python inference.py \
    --model_path /path/to/model \
    --question "Your question here" \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --top_p 0.9 \
    --top_k 50 \
    --num_beams 1
```

### Quantization

Use 8-bit or 4-bit quantization for memory efficiency:

```bash
# 8-bit quantization
python inference.py \
    --model_path /path/to/model \
    --question "Your question here" \
    --load_in_8bit

# 4-bit quantization
python inference.py \
    --model_path /path/to/model \
    --question "Your question here" \
    --load_in_4bit
```

### Display Options

Show the full prompt used for generation:

```bash
python inference.py \
    --model_path /path/to/model \
    --question "Your question here" \
    --show_prompt
```

### Programmatic Usage

Use the inference script as a module in your own code:

```python
from inference import ReasoningInference, format_output

# Initialize the inference engine
inference = ReasoningInference(
    model_path="/path/to/fine-tuned-model",
    load_in_8bit=True  # Optional: use 8-bit quantization
)

# Run inference on a single question
result = inference.infer(
    question="What is 25% of 80?",
    prompt_template="alpaca",
    max_new_tokens=512,
    temperature=0.7,
)

# Display formatted output
print(format_output(result, show_prompt=True))

# Access individual components
print("Question:", result['question'])
print("Answer:", result['answer'])
```

### Custom Prompt Creation

```python
from inference import ReasoningInference

inference = ReasoningInference(model_path="/path/to/model")

# Create a custom prompt
prompt = inference.create_prompt(
    question="Solve for x: 2x + 10 = 20",
    system_message="You are a helpful math tutor.",
    prompt_template="alpaca"
)

# Generate with the custom prompt
answer = inference.generate(
    prompt=prompt,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

print(answer)
```

### Output Format

The script produces clearly formatted output with step-by-step reasoning:

```
================================================================================
QUESTION
================================================================================
If a train travels 120 miles in 2 hours, what is its average speed?

================================================================================
STEP-BY-STEP REASONING
================================================================================

[Step 1]
To find average speed, we use the formula: speed = distance / time

[Step 2]
Given: distance = 120 miles, time = 2 hours

[Step 3]
Calculating: speed = 120 miles / 2 hours = 60 miles per hour

[Step 4]
Therefore, the train's average speed is 60 mph.

================================================================================
```

### Inference Parameters

#### Model Loading Parameters

- `--model_path`: Path to fine-tuned model (required)
- `--load_in_8bit`: Load model in 8-bit precision
- `--load_in_4bit`: Load model in 4-bit precision

#### Input Parameters

- `--question`: Single question to answer
- `--questions_file`: JSON file with multiple questions
- `--prompt_template`: Format for prompts (default, alpaca, chatml, llama2)
- `--system_message`: Custom system message

#### Generation Parameters

- `--max_new_tokens`: Maximum tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_p`: Nucleus sampling parameter (default: 0.9)
- `--top_k`: Top-k sampling parameter (default: 50)
- `--do_sample`: Use sampling (default: True)
- `--num_beams`: Number of beams for beam search (default: 1)

#### Output Parameters

- `--show_prompt`: Display the full prompt
- `--output_file`: Save results to JSON file

### Example Output Files

When using `--output_file`, results are saved in JSON format:

```json
[
  {
    "question": "What is 15% of 200?",
    "prompt": "Question: What is 15% of 200?\n\nAnswer:",
    "answer": "To find 15% of 200:\n\nStep 1: Convert percentage to decimal: 15% = 0.15\nStep 2: Multiply: 0.15 × 200 = 30\n\nTherefore, 15% of 200 is 30."
  }
]
```

## Configuration

Configurations are managed through YAML files in the `configs/` directory. Key parameters include:

- Model architecture settings
- Training hyperparameters (learning rate, batch size, etc.)
- Dataset specifications
- TPU/device configuration
- Reward function parameters

## TPU Best Practices

- Use batch sizes that are multiples of 128 for optimal TPU performance
- Leverage JAX's `pmap` for data parallelism across TPU cores
- Profile your code using JAX's profiling tools
- Monitor TPU utilization through Google Cloud Console

## Development

### Adding New Components

- **Datasets**: Add new dataset loaders to `datasets/` or extend `data/prepare_reasoning_dataset.py`
- **Reward Functions**: Implement custom reward functions in `reward_functions/`
- **Training Scripts**: Add specialized training procedures to `training/`
- **Evaluation Metrics**: Extend evaluation capabilities in `evaluation/`

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Accelerate (for multi-GPU and quantization)
- bitsandbytes (for 8-bit/4-bit quantization)

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Specify your license here]

## Citation

If you use this project in your research, please cite:

```bibtex
[Add citation information]
```

## Acknowledgments

- Built with [JAX](https://github.com/google/jax)
- TPU support via Google Cloud
- [Add other acknowledgments]

## Contact

[Add contact information]
