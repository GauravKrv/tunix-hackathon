# GRPO (Group Relative Policy Optimization) Implementation

This document describes the GRPO implementation in the Tunix training framework.

## Overview

GRPO is a policy optimization technique that ensures the model learns from within-group comparisons by:
1. Sampling multiple completions for each prompt as a group
2. Computing advantages relative to the group's mean and standard deviation
3. Using these group-relative advantages for policy gradient updates

## Key Components

### 1. Configuration (`GRPOConfig`)

```python
@dataclass
class GRPOConfig:
    num_completions_per_prompt: int = 4        # Completions per prompt
    generation_max_length: int = 256           # Max generation length
    generation_temperature: float = 0.8        # Sampling temperature
    generation_top_p: float = 0.9             # Nucleus sampling
    generation_top_k: int = 50                # Top-k sampling
    normalize_advantages: bool = True          # Normalize by std
    advantage_epsilon: float = 1e-8           # Numerical stability
    use_kl_penalty: bool = True               # KL divergence penalty
    kl_penalty_coef: float = 0.01            # KL penalty weight
```

### 2. Group Sampling (`sample_completions_for_prompts`)

For each prompt in the batch:
- Generates N completions using `model.generate()`
- Returns completions, attention masks, and log probabilities
- Flattens into shape: `(batch_size * N, sequence_length)`

### 3. Advantage Calculation (`compute_group_advantages`)

```python
# Reshape rewards: (batch_size * N,) -> (batch_size, N)
rewards_grouped = rewards.view(batch_size, num_completions_per_prompt)

# Compute group statistics
group_means = rewards_grouped.mean(dim=1, keepdim=True)
group_stds = rewards_grouped.std(dim=1, keepdim=True)

# Normalize advantages
if normalize_advantages:
    advantages = (rewards_grouped - group_means) / (group_stds + epsilon)
else:
    advantages = rewards_grouped - group_means

# Flatten back: (batch_size, N) -> (batch_size * N,)
advantages = advantages.view(-1)
```

### 4. Policy Loss (`grpo_training_step`)

```python
# Compute policy loss with group-relative advantages
policy_loss = -(advantages.detach() * sequence_log_probs).mean()

# Optional KL penalty
if use_kl_penalty:
    kl_div = (old_log_probs.detach() - sequence_log_probs).abs()
    total_loss = policy_loss + kl_penalty_coef * kl_div.mean()
```

## Validation Checks

The implementation includes extensive validation to ensure correctness:

### Pre-training Validation (`validate_grpo_setup`)
- ✅ `num_completions_per_prompt > 1`
- ✅ Tokenizer is available for generation
- ✅ Dataset returns prompts only (not completions)

### Runtime Validation (`grpo_training_step`)
- ✅ Correct number of completions sampled: `batch_size * N`
- ✅ Advantage shape matches: `(batch_size, N)`
- ✅ Advantages centered per group: `mean ≈ 0` (when normalized)
- ✅ Logs first 3 groups with rewards and advantages

### Assertion Examples

```python
# Verify completion count
assert completion_ids.size(0) == batch_size * num_completions, \
    f"Expected {batch_size * num_completions} completions, got {completion_ids.size(0)}"

# Verify advantage shape
assert advantages_grouped.shape == (batch_size, num_completions), \
    f"Advantages shape mismatch: expected ({batch_size}, {num_completions})"

# Verify advantages are centered
for group_idx in range(batch_size):
    group_mean = advantages_grouped[group_idx].mean()
    assert abs(group_mean) < 1e-5 or not normalize_advantages, \
        f"Group {group_idx} advantages not centered: mean={group_mean:.6f}"
```

## Usage

### Command Line

```bash
# Enable GRPO with default settings
python train.py --use_grpo

# Customize GRPO parameters
python train.py --use_grpo \
    --num_completions_per_prompt 4 \
    --generation_max_length 256 \
    --batch_size 4 \
    --learning_rate 5e-5
```

### Python API

```python
from train import TunixConfig, TrainingConfig, GRPOConfig

config = TunixConfig(
    training=TrainingConfig(
        use_grpo=True,
        batch_size=4,
    ),
    grpo=GRPOConfig(
        num_completions_per_prompt=4,
        generation_max_length=256,
        normalize_advantages=True,
    ),
)
```

## Logging Output

### GRPO Validation Logs

```
============================================================
GRPO CONFIGURATION VALIDATED
============================================================
Number of completions per prompt: 4
Generation max length: 256
Generation temperature: 0.8
Normalize advantages: True
Use KL penalty: True
============================================================
```

### Training Step Logs

```
GRPO Validation: Sampled 16 completions for 4 prompts (4 per prompt)
GRPO Validation: Advantage stats per group - mean: 0.0234, std: 0.8912
GRPO Group 0: rewards=[0.56, 0.72, 0.48, 0.65], advantages=[-0.45, 0.89, -1.12, 0.23]
GRPO Validation: All assertions passed - group sampling verified

Epoch: 1 | Step: 10 | Policy Loss: 2.3456 | Reward: 0.6234 | Advantage: 0.0012 | LR: 5.00e-05
```

## Design Rationale

### Why Group-Relative Advantages?

Traditional RL treats each sample independently, which can lead to:
- High variance in gradient estimates
- Difficulty distinguishing good from bad outputs
- Instability during training

GRPO addresses this by:
- **Relative Comparison**: Each completion is evaluated relative to its prompt's group
- **Variance Reduction**: Normalization by group std reduces gradient variance
- **Fair Comparison**: Only compares completions from the same prompt
- **Stable Learning**: Group statistics provide more reliable signals

### Key Differences from Standard Policy Gradient

| Aspect | Standard PG | GRPO |
|--------|------------|------|
| Reward baseline | Global mean | Group mean |
| Normalization | Global std | Group std |
| Comparison | All samples | Within-group only |
| Variance | Higher | Lower |
| Sample efficiency | Lower | Higher |

## Example Output Structure

For a batch with 4 prompts and 4 completions per prompt:

```
Input:
  Prompt 1, Prompt 2, Prompt 3, Prompt 4
  Shape: (4, seq_len)

After Sampling:
  P1_C1, P1_C2, P1_C3, P1_C4, P2_C1, P2_C2, P2_C3, P2_C4, ...
  Shape: (16, seq_len)

After Grouping:
  [[P1_C1, P1_C2, P1_C3, P1_C4],
   [P2_C1, P2_C2, P2_C3, P2_C4],
   [P3_C1, P3_C2, P3_C3, P3_C4],
   [P4_C1, P4_C2, P4_C3, P4_C4]]
  Shape: (4, 4)

Advantages (per group):
  Group 1: [-0.5, 1.2, -0.3, 0.1]  (mean ≈ 0, std ≈ 1)
  Group 2: [0.8, -0.6, 0.2, -0.4]  (mean ≈ 0, std ≈ 1)
  ...
```

## Troubleshooting

### Common Issues

1. **"GRPO requires num_completions_per_prompt > 1"**
   - Solution: Set `--num_completions_per_prompt` to at least 2

2. **"GRPO requires dataset with return_prompts_only=True"**
   - Solution: Ensure dataset is created with `return_prompts_only=True` when `use_grpo=True`

3. **"Group advantages not centered"**
   - This indicates a bug in advantage calculation
   - Check that `normalize_advantages=True` in config

4. **OOM (Out of Memory)**
   - Reduce `num_completions_per_prompt`
   - Reduce `batch_size`
   - Reduce `generation_max_length`

## Performance Considerations

- **Memory**: GRPO requires `N * batch_size` forward passes for generation
- **Compute**: Generation is done with `torch.no_grad()` to save memory
- **Trade-off**: More completions per prompt → better gradients but slower training
- **Recommended**: Start with `num_completions_per_prompt=4` and adjust based on resources

## References

The GRPO implementation is based on group-relative policy optimization principles, which normalize rewards within groups rather than globally. This approach is particularly effective for:
- Text generation tasks where prompts vary widely
- Environments with high reward variance
- Scenarios requiring sample-efficient learning
