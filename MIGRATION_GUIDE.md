# Migration Guide: Class-Based to Function-Based Rewards

This guide helps you migrate code from the old class-based reward system to the new pure function-based system.

## Quick Reference

### Import Changes

```python
# OLD
from rewards import (
    RewardFunction,
    CorrectnessReward,
    ReasoningCoherenceReward,
    ExplanationQualityReward,
    CompositeReward,
    CompositionStrategy
)

# NEW
from rewards import (
    correctness_reward,
    structure_reward,
    conciseness_reward,
    compute_composite_reward,
    RewardComponent
)
```

## Migration Examples

### Example 1: Basic Correctness Reward

```python
# OLD
config = {'exact_match_weight': 0.6, 'partial_match_weight': 0.4}
correctness = CorrectnessReward(config)
score = correctness.compute(prediction="42", ground_truth="42")

# NEW
score = correctness_reward(
    prediction="42",
    ground_truth="42",
    exact_match_weight=0.6,
    partial_match_weight=0.4
)
```

### Example 2: Structure/Coherence Reward

```python
# OLD
config = {
    'min_steps': 1,
    'max_steps': 10,
    'logical_flow_weight': 0.5,
    'step_quality_weight': 0.5
}
coherence = ReasoningCoherenceReward(config)
score = coherence.compute(reasoning_steps=["step1", "step2", "step3"])

# NEW
score = structure_reward(
    reasoning_steps=["step1", "step2", "step3"],
    min_steps=1,
    max_steps=10,
    logical_flow_weight=0.5,
    step_quality_weight=0.5
)
```

### Example 3: Conciseness/Quality Reward

```python
# OLD
config = {
    'min_length': 50,
    'max_length': 1000,
    'clarity_weight': 0.4,
    'completeness_weight': 0.3,
    'structure_weight': 0.3
}
quality = ExplanationQualityReward(config)
score = quality.compute(explanation="This is my explanation...")

# NEW
score = conciseness_reward(
    explanation="This is my explanation...",
    min_length=50,
    max_length=1000,
    clarity_weight=0.4,
    completeness_weight=0.3,
    structure_weight=0.3
)
```

### Example 4: Simple Additive Composite Reward

```python
# OLD
from rewards import create_simple_additive_reward

reward_functions = {
    'correctness': CorrectnessReward().compute,
    'coherence': ReasoningCoherenceReward().compute
}
weights = {'correctness': 0.6, 'coherence': 0.4}
composite = create_simple_additive_reward(reward_functions, weights)
score = composite(prediction="42", ground_truth="42", reasoning_steps=["step1"])

# NEW
from rewards import (
    correctness_reward,
    structure_reward,
    create_reward_components,
    compute_composite_reward
)

reward_functions = {
    'correctness': correctness_reward,
    'structure': structure_reward
}
weights = {'correctness': 0.6, 'structure': 0.4}
components = create_reward_components(reward_functions, weights)
score = compute_composite_reward(
    components,
    prediction="42",
    ground_truth="42",
    reasoning_steps=["step1"]
)
```

### Example 5: Composite Reward with Component Breakdown

```python
# OLD
composite = CompositeReward(
    components=[...],
    strategy=CompositionStrategy.ADDITIVE,
    return_components=True
)
total_score, component_scores = composite(
    prediction="42",
    ground_truth="42",
    reasoning_steps=["step1"]
)

# NEW
from rewards import compute_composite_reward_with_details

total_score, component_scores = compute_composite_reward_with_details(
    components,
    prediction="42",
    ground_truth="42",
    reasoning_steps=["step1"]
)
```

### Example 6: Multiplicative Composition (NO LONGER SUPPORTED)

```python
# OLD
composite = CompositeReward(
    components=[...],
    strategy=CompositionStrategy.MULTIPLICATIVE
)
score = composite(...)

# NEW - Use weighted summation instead
# If you need multiplicative behavior, you must implement it manually:
from rewards import correctness_reward, structure_reward

c_score = correctness_reward(prediction="42", ground_truth="42")
s_score = structure_reward(reasoning_steps=["step1"])

# Manual multiplicative composition (if absolutely needed)
multiplicative_score = (c_score ** 0.6) * (s_score ** 0.4)

# Recommended: Use weighted summation
additive_score = 0.6 * c_score + 0.4 * s_score
```

## Common Patterns

### Pattern 1: Reusable Component Configuration

```python
# Create reusable component definitions
components = [
    RewardComponent(
        name="correctness",
        reward_fn=correctness_reward,
        weight=0.5,
        clip_range=(0.0, 1.0),
        normalize=True
    ),
    RewardComponent(
        name="structure",
        reward_fn=structure_reward,
        weight=0.3,
        clip_range=(0.0, 1.0),
        normalize=True
    ),
    RewardComponent(
        name="conciseness",
        reward_fn=conciseness_reward,
        weight=0.2,
        clip_range=(0.0, 1.0),
        normalize=True
    )
]

# Use in multiple places
score1 = compute_composite_reward(components, **data1)
score2 = compute_composite_reward(components, **data2)
```

### Pattern 2: Custom Reward Function

```python
# Define your own reward function
def custom_length_reward(text: str, target_length: int = 100, **kwargs) -> float:
    """Reward based on proximity to target length."""
    actual_length = len(text.split())
    diff = abs(actual_length - target_length)
    return max(0.0, 1.0 - diff / target_length)

# Use it with the system
component = RewardComponent(
    name="custom_length",
    reward_fn=custom_length_reward,
    weight=0.25
)

components = [
    component,
    RewardComponent(name="correctness", reward_fn=correctness_reward, weight=0.75)
]

score = compute_composite_reward(
    components,
    text="some text here",
    target_length=50,
    prediction="42",
    ground_truth="42"
)
```

### Pattern 3: Dynamic Weight Adjustment

```python
# Start with default weights
weights = {"correctness": 0.5, "structure": 0.3, "conciseness": 0.2}
components = create_reward_components(
    {
        "correctness": correctness_reward,
        "structure": structure_reward,
        "conciseness": conciseness_reward
    },
    weights
)

# Later, adjust weights dynamically
for component in components:
    if component.name == "correctness":
        component.weight = 0.7  # Emphasize correctness more
    elif component.name == "structure":
        component.weight = 0.2
    elif component.name == "conciseness":
        component.weight = 0.1

# Use updated components
score = compute_composite_reward(components, **data)
```

## Breaking Changes Summary

1. **No Base Class**: `RewardFunction` base class removed
2. **No Class Instantiation**: Use functions directly
3. **No Multiplicative Composition**: Only weighted summation supported
4. **No Adaptive Rewards**: `AdaptiveCompositeReward` removed
5. **Function Renaming**:
   - `ReasoningCoherenceReward` → `structure_reward`
   - `ExplanationQualityReward` → `conciseness_reward`
6. **Configuration Changes**: Pass parameters directly instead of config dicts

## Benefits of Migration

1. **Simpler Code**: No need to instantiate classes
2. **More Testable**: Pure functions are easier to test
3. **Better Performance**: No object overhead
4. **Clearer Intent**: Function names directly describe what they do
5. **Easier Debugging**: Weighted summation is transparent

## Getting Help

- See `rewards_example.py` for complete examples
- Read `rewards/README.md` for detailed documentation
- Check `REFACTORING_SUMMARY.md` for technical details
