# Examples

This directory contains example configurations, recipes, and notebooks for training and inference.

## Structure

```
examples/
├── configs/           # Configuration files for different setups
│   ├── datasets/      # Dataset-specific configurations
│   ├── models/        # Model-specific configurations
│   └── rewards/       # Reward function compositions
├── recipes/           # Complete training recipes
├── notebooks/         # Jupyter notebooks for tutorials
└── README.md
```

## Quick Start

The fastest way to get started is with the quickstart notebook:

```bash
jupyter notebook examples/notebooks/quickstart.ipynb
```

## Configurations

### Datasets
- `configs/datasets/anthropic_hh.yaml` - Anthropic Helpful & Harmless dataset
- `configs/datasets/openassistant.yaml` - OpenAssistant dataset
- `configs/datasets/summarization.yaml` - Summarization dataset (TL;DR, etc.)
- `configs/datasets/custom.yaml` - Template for custom datasets

### Models
- `configs/models/gemma2_2b.yaml` - Gemma2 2B configuration
- `configs/models/gemma3_1b.yaml` - Gemma3 1B configuration

### Reward Functions
- `configs/rewards/single_reward.yaml` - Single reward model
- `configs/rewards/multi_objective.yaml` - Multiple reward objectives
- `configs/rewards/ensemble.yaml` - Ensemble of reward models
- `configs/rewards/rule_based_hybrid.yaml` - Hybrid rule-based + learned rewards

## Recipes

### Basic Training
- `recipes/gemma2_2b_basic.yaml` - Basic training with Gemma2 2B
- `recipes/gemma3_1b_basic.yaml` - Basic training with Gemma3 1B

### Advanced Training
- `recipes/gemma2_2b_multi_reward.yaml` - Multi-objective reward training
- `recipes/gemma3_1b_ensemble.yaml` - Ensemble reward training
- `recipes/gemma2_2b_large_batch.yaml` - Large batch training for better stability

### Dataset-Specific
- `recipes/anthropic_hh_gemma2.yaml` - Anthropic HH with Gemma2 2B
- `recipes/openassistant_gemma3.yaml` - OpenAssistant with Gemma3 1B
- `recipes/summarization_gemma2.yaml` - Summarization task with Gemma2 2B

## Usage

### Using a Recipe

```bash
python train.py --config examples/recipes/gemma2_2b_basic.yaml
```

### Composing Custom Configuration

```bash
python train.py \
  --config examples/configs/models/gemma2_2b.yaml \
  --config examples/configs/datasets/anthropic_hh.yaml \
  --config examples/configs/rewards/multi_objective.yaml
```

### Running Inference

```python
from inference import load_model, generate

model = load_model("checkpoints/gemma2_2b_final")
response = generate(model, "Hello, how are you?")
print(response)
```

See the quickstart notebook for more detailed examples.
