# Reward Functions

This module provides pure function-based reward functions for training and evaluating language models. All reward functions are standalone functions without class hierarchies or state management.

## Design Philosophy

- **Pure Functions**: All reward functions are stateless and side-effect-free
- **No Inheritance**: No base classes or object-oriented hierarchies
- **Weighted Summation Only**: Composite rewards use simple weighted addition (w1*r1 + w2*r2 + w3*r3)
- **Composable**: Functions can be easily combined using `RewardComponent` wrappers

## Core Reward Functions

### 1. Correctness Reward

Evaluates the accuracy of predictions against ground truth.

```python
from rewards import correctness_reward

score = correctness_reward(
    prediction="42",
    ground_truth="42",
    exact_match_weight=0.6,
    partial_match_weight=0.4
)
```

**Parameters:**
- `prediction`: Model prediction (any type, converted to string)
- `ground_truth`: Ground truth answer (any type, converted to string)
- `exact_match_weight`: Weight for exact match component (default: 0.6)
- `partial_match_weight`: Weight for partial match component (default: 0.4)

**Returns:** Float score between 0.0 and 1.0

### 2. Structure Reward

Evaluates reasoning coherence and step quality.

```python
from rewards import structure_reward

score = structure_reward(
    reasoning_steps=["Step 1", "Therefore, step 2", "Thus, conclusion"],
    min_steps=1,
    max_steps=10,
    logical_flow_weight=0.5,
    step_quality_weight=0.5
)
```

**Parameters:**
- `reasoning_steps`: List of reasoning step strings
- `min_steps`: Minimum expected number of steps (default: 1)
- `max_steps`: Maximum expected number of steps (default: 10)
- `logical_flow_weight`: Weight for logical flow score (default: 0.5)
- `step_quality_weight`: Weight for step quality score (default: 0.5)
- `min_step_length`: Minimum expected length per step (default: 10)

**Returns:** Float score between 0.0 and 1.0

### 3. Conciseness Reward

Evaluates explanation quality, clarity, and conciseness.

```python
from rewards import conciseness_reward

score = conciseness_reward(
    explanation="This is a clear and complete explanation.",
    min_length=50,
    max_length=1000,
    clarity_weight=0.4,
    completeness_weight=0.3,
    structure_weight=0.3
)
```

**Parameters:**
- `explanation`: The explanation text to evaluate
- `min_length`: Minimum expected length (default: 50)
- `max_length`: Maximum expected length (default: 1000)
- `clarity_weight`: Weight for clarity score (default: 0.4)
- `completeness_weight`: Weight for completeness score (default: 0.3)
- `structure_weight`: Weight for structure score (default: 0.3)

**Returns:** Float score between 0.0 and 1.0

## Composite Rewards

### Simple Weighted Summation

```python
from rewards import compute_composite_reward, RewardComponent, correctness_reward, structure_reward

components = [
    RewardComponent(name="correctness", reward_fn=correctness_reward, weight=0.6),
    RewardComponent(name="structure", reward_fn=structure_reward, weight=0.4)
]

total_score = compute_composite_reward(
    components,
    prediction="42",
    ground_truth="42",
    reasoning_steps=["Step 1", "Step 2"]
)
```

### With Component Details

```python
from rewards import compute_composite_reward_with_details, create_reward_components

reward_functions = {
    "correctness": correctness_reward,
    "structure": structure_reward,
    "conciseness": conciseness_reward
}

weights = {
    "correctness": 0.5,
    "structure": 0.3,
    "conciseness": 0.2
}

components = create_reward_components(reward_functions, weights)

total_score, component_scores = compute_composite_reward_with_details(
    components,
    prediction="42",
    ground_truth="42",
    reasoning_steps=["Step 1", "Step 2"],
    explanation="Clear explanation"
)

print(f"Total: {total_score}")
print(f"Components: {component_scores}")
```

## Process Reward Modeling

For scoring reasoning trajectories with multiple steps:

```python
from rewards import (
    ProcessStep,
    score_reasoning_step,
    score_reasoning_trajectory,
    RewardComponent,
    structure_reward
)

# Create process steps
steps = [
    ProcessStep(step_index=0, content="First, identify the problem", action="analyze"),
    ProcessStep(step_index=1, content="Then, apply the formula", action="compute"),
    ProcessStep(step_index=2, content="Finally, verify the result", action="verify")
]

# Define reward components for steps
step_components = [
    RewardComponent(name="structure", reward_fn=structure_reward, weight=1.0)
]

# Score the trajectory
result = score_reasoning_trajectory(
    steps=steps,
    step_reward_components=step_components,
    discount_factor=1.0
)

print(f"Aggregate score: {result.aggregate_score}")
print(f"Individual steps: {result.step_scores}")
```

## Key Features

1. **No Multiplicative Composition**: Only weighted summation is supported
   - Formula: `total = w1*r1 + w2*r2 + w3*r3`
   - No geometric means, no multiplicative combinations

2. **Stateless Functions**: Every function call is independent
   - No hidden state or side effects
   - Same inputs always produce same outputs

3. **Optional Clipping/Normalization**: RewardComponent supports:
   ```python
   component = RewardComponent(
       name="my_reward",
       reward_fn=correctness_reward,
       weight=0.5,
       clip_range=(0.0, 1.0),  # Clip to [0, 1]
       normalize=True           # Normalize after clipping
   )
   ```

4. **Flexible Composition**: Combine any reward functions with any weights
   ```python
   # Equal weighting
   weights = {"r1": 0.33, "r2": 0.33, "r3": 0.34}
   
   # Emphasize correctness
   weights = {"correctness": 0.7, "structure": 0.2, "conciseness": 0.1}
   ```

## Migration from Class-Based Design

**Before (class-based):**
```python
from rewards import CorrectnessReward, CompositeReward

correctness = CorrectnessReward(config={'exact_match_weight': 0.6})
score = correctness.compute(prediction="42", ground_truth="42")

composite = CompositeReward(components, strategy=CompositionStrategy.MULTIPLICATIVE)
total = composite(prediction="42", ground_truth="42")
```

**After (function-based):**
```python
from rewards import correctness_reward, compute_composite_reward, RewardComponent

score = correctness_reward(
    prediction="42",
    ground_truth="42",
    exact_match_weight=0.6
)

components = [RewardComponent(name="c", reward_fn=correctness_reward, weight=1.0)]
total = compute_composite_reward(components, prediction="42", ground_truth="42")
```

## Benefits

1. **Simpler**: No class hierarchies, no inheritance, no `__init__` methods
2. **Testable**: Pure functions are easy to test
3. **Composable**: Mix and match functions easily
4. **Predictable**: No hidden state, no side effects
5. **Transparent**: Weighted summation is easy to understand and debug

## Example Usage

See `rewards_example.py` in the project root for comprehensive examples.
