# GRPO Training Flow

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                        │
└─────────────────────────────────────────────────────────────┘

1. SETUP PHASE
   ├── Load model and tokenizer
   ├── Create dataset with return_prompts_only=True
   ├── Validate GRPO configuration
   └── ✅ Assert: num_completions_per_prompt > 1

2. TRAINING STEP
   ├── Receive batch of prompts
   │   Shape: (batch_size, seq_len)
   │
   ├── SAMPLING PHASE (sample_completions_for_prompts)
   │   ├── For each prompt, generate N completions
   │   ├── Use model.generate() with temperature sampling
   │   ├── Compute log probabilities for each completion
   │   └── ✅ Assert: completion_ids.size(0) == batch_size * N
   │       Output: (batch_size * N, seq_len)
   │
   ├── REWARD PHASE (compute_rewards_for_completions)
   │   ├── Forward pass through model for all completions
   │   ├── Compute composite reward (quality, safety, diversity, coherence)
   │   └── Output: rewards tensor of shape (batch_size * N,)
   │
   ├── ADVANTAGE PHASE (compute_group_advantages)
   │   ├── Reshape: (batch_size * N,) -> (batch_size, N)
   │   ├── Compute group statistics:
   │   │   ├── group_means = rewards_grouped.mean(dim=1)
   │   │   └── group_stds = rewards_grouped.std(dim=1)
   │   ├── Normalize advantages:
   │   │   └── advantages = (rewards - means) / (stds + epsilon)
   │   ├── ✅ Assert: advantages_grouped.shape == (batch_size, N)
   │   ├── ✅ Assert: Each group mean ≈ 0 (if normalized)
   │   └── Reshape back: (batch_size, N) -> (batch_size * N,)
   │
   ├── POLICY LOSS PHASE
   │   ├── Forward pass to get current log probs
   │   ├── Compute policy loss:
   │   │   └── loss = -mean(advantages.detach() * log_probs)
   │   ├── Optional KL penalty:
   │   │   └── kl = kl_coef * |old_log_probs - log_probs|
   │   └── ✅ Validation logs: Group statistics
   │
   └── GRADIENT UPDATE
       ├── Backward pass
       ├── Gradient clipping
       └── Optimizer step

3. LOGGING
   ├── Policy loss
   ├── Mean reward per group
   ├── Mean advantage per group
   └── Advantage std per group
```

## Data Flow

```
Input Prompts:
┌─────────────┐
│ Prompt 1    │
│ Prompt 2    │
│ Prompt 3    │
│ Prompt 4    │
└─────────────┘
   (4, seq_len)

      ↓ sample_completions_for_prompts (N=4)

Completions:
┌─────────────────────────────────────────┐
│ P1_C1, P1_C2, P1_C3, P1_C4              │
│ P2_C1, P2_C2, P2_C3, P2_C4              │
│ P3_C1, P3_C2, P3_C3, P3_C4              │
│ P4_C1, P4_C2, P4_C3, P4_C4              │
└─────────────────────────────────────────┘
   (16, seq_len)

      ↓ compute_rewards_for_completions

Rewards (flat):
┌────────────────────────────────────────┐
│ r1, r2, r3, r4, r5, r6, r7, r8, ...    │
└────────────────────────────────────────┘
   (16,)

      ↓ compute_group_advantages (reshape)

Rewards (grouped):
┌──────────────────────┐
│ [r1,  r2,  r3,  r4]  │  Group 1 (Prompt 1)
│ [r5,  r6,  r7,  r8]  │  Group 2 (Prompt 2)
│ [r9,  r10, r11, r12] │  Group 3 (Prompt 3)
│ [r13, r14, r15, r16] │  Group 4 (Prompt 4)
└──────────────────────┘
   (4, 4)

      ↓ normalize per group

Advantages (grouped):
┌───────────────────────────┐
│ [-0.5,  1.2, -0.3,  0.1]  │  mean ≈ 0, std ≈ 1
│ [ 0.8, -0.6,  0.2, -0.4]  │  mean ≈ 0, std ≈ 1
│ [-1.0,  0.7,  0.5, -0.2]  │  mean ≈ 0, std ≈ 1
│ [ 0.3, -0.9,  0.8, -0.2]  │  mean ≈ 0, std ≈ 1
└───────────────────────────┘
   (4, 4)

      ↓ flatten

Advantages (flat):
┌──────────────────────────────────────────┐
│ -0.5, 1.2, -0.3, 0.1, 0.8, -0.6, ...     │
└──────────────────────────────────────────┘
   (16,)

      ↓ policy gradient

Loss = -mean(advantages * log_probs)
```

## Validation Checkpoints

```
┌─────────────────────────────────────────┐
│    Validation Checkpoint Timeline        │
└─────────────────────────────────────────┘

BEFORE TRAINING:
✅ validate_grpo_setup()
   ├── num_completions_per_prompt > 1
   ├── tokenizer is not None
   └── dataset.return_prompts_only == True

DURING SAMPLING:
✅ sample_completions_for_prompts()
   └── completion_ids.size(0) == batch_size * N

AFTER ADVANTAGE CALCULATION:
✅ compute_group_advantages()
   ├── rewards divisible by num_completions
   └── returns flattened advantages

BEFORE GRADIENT UPDATE:
✅ grpo_training_step()
   ├── advantages_grouped.shape == (batch_size, N)
   ├── each group mean ≈ 0 (if normalized)
   └── log first 3 groups for inspection

AFTER GRADIENT UPDATE:
✅ Log metrics
   ├── policy_loss
   ├── mean_reward
   ├── mean_advantage
   └── std_advantage
```

## Key Properties Maintained

### Group Structure Invariants

1. **Size Invariant**: 
   ```
   total_completions = batch_size * num_completions_per_prompt
   ```

2. **Group Invariant**: 
   ```
   For each group i:
     - Contains exactly N completions
     - All from the same prompt
     - Advantages sum to ≈ 0 (if normalized)
   ```

3. **Shape Invariant**:
   ```
   rewards.shape: (batch_size * N,)
   ↓ view
   rewards_grouped.shape: (batch_size, N)
   ↓ normalize
   advantages_grouped.shape: (batch_size, N)
   ↓ view
   advantages.shape: (batch_size * N,)
   ```

## Advantage Calculation Detail

```python
# For each group independently:
for group_idx in range(batch_size):
    # Extract group rewards
    group_rewards = rewards_grouped[group_idx]  # Shape: (N,)
    
    # Compute group statistics
    group_mean = group_rewards.mean()           # Scalar
    group_std = group_rewards.std()             # Scalar
    
    # Normalize
    group_advantages = (group_rewards - group_mean) / (group_std + epsilon)
    
    # Verify: group_advantages.mean() ≈ 0
    # Verify: group_advantages.std() ≈ 1
```

## Example Execution Trace

```
Step 1: Load batch of 4 prompts
  Prompts: ["What is AI?", "Explain GPT", "Define ML", "What is DL?"]

Step 2: Sample 4 completions per prompt
  Total completions: 16
  ✅ Assert: 16 == 4 * 4

Step 3: Compute rewards
  Rewards: [0.6, 0.8, 0.5, 0.7, 0.9, 0.3, 0.6, 0.8, ...]
  
Step 4: Group rewards
  Group 0: [0.6, 0.8, 0.5, 0.7]  mean=0.65, std=0.11
  Group 1: [0.9, 0.3, 0.6, 0.8]  mean=0.65, std=0.25
  Group 2: [0.7, 0.5, 0.9, 0.6]  mean=0.68, std=0.16
  Group 3: [0.4, 0.8, 0.6, 0.5]  mean=0.58, std=0.15

Step 5: Compute advantages
  Group 0: [-0.45, 1.36, -1.36, 0.45]  mean≈0, std≈1
  Group 1: [1.00, -1.40, -0.20, 0.60]  mean≈0, std≈1
  Group 2: [0.12, -1.12, 1.38, -0.38]  mean≈0, std≈1
  Group 3: [-1.20, 1.47, 0.13, -0.40]  mean≈0, std≈1
  ✅ Assert: Each group mean ≈ 0

Step 6: Compute policy loss
  loss = -mean(advantages * log_probs)
  loss = -mean([-0.45*(-2.3), 1.36*(-1.8), ...])
  
Step 7: Backward and update
  loss.backward()
  optimizer.step()
```
