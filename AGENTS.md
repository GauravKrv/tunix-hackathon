# Agent Development Guide

## Setup
```bash
pip install -r requirements.txt
```

## Commands
- **Build**: N/A (Python project, no build required)
- **Lint**: `pylint train.py utils.py example_usage.py` or `flake8 *.py`
- **Test**: N/A (test suite not yet implemented)
- **Dev Server**: N/A
- **Training**: `python train.py --output_dir ./outputs --num_epochs 3`
- **GRPO Training**: `python train.py --output_dir ./outputs --use_grpo --num_completions_per_prompt 4 --generation_max_length 256`
- **TPU Training**: `./run_tpu_training.sh`
- **Inference**: `python inference.py --model_path /path/to/model --question "Your question"`

## Tech Stack
- Python 3.8+
- PyTorch 2.1.0+ with XLA for TPU support
- Transformers 4.36.0+ for Gemma2 model
- torch-xla for TPU optimization

## Architecture
- `train.py`: Main training script with TunixTrainer, GRPO implementation, composite reward functions, and TPU optimizations
- `utils.py`: Utility functions for configuration, checkpointing, and metrics
- `example_usage.py`: Example scripts demonstrating custom configurations including GRPO
- `rewards_example.py`: Example demonstrating pure function-based reward system
- `run_tpu_training.sh`: Shell script for easy TPU training execution
- `inference.py`: Inference script for fine-tuned models with step-by-step reasoning
- `requirements.txt`: Python dependencies
- `config_example.json`: Example configuration file
- `rewards/`: Pure function-based reward system
  - `correctness_reward.py`: Standalone correctness reward function
  - `reasoning_coherence_reward.py`: Standalone structure reward function
  - `explanation_quality_reward.py`: Standalone conciseness reward function
  - `composite_reward.py`: Weighted summation utilities for combining rewards

## GRPO (Group Relative Policy Optimization)

GRPO is a policy optimization technique that samples multiple completions for each prompt and computes advantages relative to the group mean and standard deviation. This allows the model to learn from within-group comparisons rather than treating samples independently.

### Key Features

1. **Group-based Sampling**: For each prompt, multiple completions are sampled as a group
2. **Group-relative Advantages**: Advantages are normalized using the mean and std of rewards within each group
3. **Policy Gradient Updates**: Uses group-relative advantages for gradient computation
4. **Validation Checks**: Assertions ensure group sampling is occurring before gradients are computed

### GRPO Configuration

```python
GRPOConfig(
    num_completions_per_prompt=4,      # Number of completions per prompt
    generation_max_length=256,         # Max length for generated completions
    generation_temperature=0.8,        # Sampling temperature
    generation_top_p=0.9,              # Nucleus sampling parameter
    generation_top_k=50,               # Top-k sampling parameter
    normalize_advantages=True,         # Normalize advantages by std
    advantage_epsilon=1e-8,            # Epsilon for numerical stability
    use_kl_penalty=True,               # Apply KL divergence penalty
    kl_penalty_coef=0.01,             # KL penalty coefficient
)
```

### Usage Example

```bash
# Enable GRPO training with 4 completions per prompt
python train.py --use_grpo --num_completions_per_prompt 4 --generation_max_length 256

# Or use the example script
python example_usage.py 3
```

### How It Works

1. **Sampling Phase**: For each prompt in the batch, sample N completions
2. **Reward Computation**: Compute rewards for all completions using the composite reward function
3. **Advantage Calculation**: For each group (prompt), compute advantages as:
   - `advantages = (rewards - group_mean) / (group_std + epsilon)`
4. **Policy Update**: Use advantages to weight the policy gradient:
   - `loss = -mean(advantages * log_probs)`
5. **Validation**: Assert that advantages are correctly centered within each group

### Validation Checks

The implementation includes multiple validation checks:
- Ensures `num_completions_per_prompt > 1`
- Verifies correct number of completions sampled
- Checks advantage shapes match expected dimensions
- Validates advantages are centered within each group (mean ≈ 0)
- Logs detailed statistics per group for debugging

## Code Style
- Follow PEP 8 conventions
- Use type hints for function signatures
- Use dataclasses for configuration objects
- Keep functions focused and modular
- Add docstrings to classes and public methods

## Implementation Summary

### GRPO Implementation Details

The GRPO implementation modifies the training loop to:

1. **Dataset Mode**: When `use_grpo=True`, datasets return prompts only (without completions)
2. **Sampling**: In `sample_completions_for_prompts()`, the model generates N completions per prompt
3. **Reward Calculation**: All completions receive rewards from the composite reward function
4. **Group Advantages**: In `compute_group_advantages()`, rewards are grouped by prompt and normalized:
   - Reshape rewards into (batch_size, num_completions_per_prompt)
   - Compute mean and std per group
   - Normalize: `(reward - mean) / (std + epsilon)`
5. **Policy Loss**: Advantages are detached and used to weight log probabilities
6. **Assertions**: Multiple checks ensure group structure is maintained throughout

### Key Methods

- `sample_completions_for_prompts()`: Generates multiple completions per prompt using model.generate()
- `compute_rewards_for_completions()`: Computes rewards for all generated completions
- `compute_group_advantages()`: Calculates group-relative advantages with normalization
- `grpo_training_step()`: Main GRPO training step with validation checks
- `validate_grpo_setup()`: Pre-training validation of GRPO configuration
