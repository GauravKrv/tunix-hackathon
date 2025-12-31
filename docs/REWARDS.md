# Reward Function Design and Tuning

This guide covers the theory, implementation, and tuning of reward functions for reinforcement learning.

## Table of Contents

- [Introduction](#introduction)
- [Reward Function Theory](#reward-function-theory)
- [Designing Reward Functions](#designing-reward-functions)
- [Implementation Guide](#implementation-guide)
- [Tuning Strategies](#tuning-strategies)
- [Common Patterns](#common-patterns)
- [Debugging Reward Functions](#debugging-reward-functions)
- [Best Practices](#best-practices)

## Introduction

The reward function is the most critical component of reinforcement learning systems. It defines the objective the agent learns to optimize and directly impacts training stability, convergence speed, and final policy quality.

### Key Principles

1. **Alignment**: Reward must align with desired behavior
2. **Density**: Provide frequent feedback to guide learning
3. **Scaling**: Maintain appropriate magnitude for stable learning
4. **Consistency**: Avoid contradictory or ambiguous signals
5. **Simplicity**: Start simple, add complexity only when needed

## Reward Function Theory

### Basic Components

A reward function `R(s, a, s')` maps state-action-next_state tuples to scalar values:

```
R: S × A × S → ℝ
```

Where:
- `s` is the current state
- `a` is the action taken
- `s'` is the resulting next state
- Return value is the immediate reward

### Types of Rewards

#### 1. Sparse Rewards

Rewards given only at episode completion or goal achievement.

**Pros:**
- Simple to define
- Clear objective
- Less prone to reward hacking

**Cons:**
- Slow learning
- Requires extensive exploration
- May never discover positive rewards

**Example:**
```python
def sparse_reward(state, action, next_state, done, info):
    """Return reward only on goal achievement."""
    if done and info.get('success', False):
        return 1.0
    return 0.0
```

#### 2. Dense Rewards

Frequent feedback at every step to guide learning.

**Pros:**
- Faster learning
- Guides exploration
- Better for complex tasks

**Cons:**
- Risk of reward hacking
- May learn suboptimal policies
- Harder to design correctly

**Example:**
```python
def dense_reward(state, action, next_state, done, info):
    """Provide feedback at every step."""
    goal_position = info['goal_position']
    distance = np.linalg.norm(next_state['position'] - goal_position)
    return -distance  # Negative distance as reward
```

#### 3. Hybrid Rewards

Combination of sparse and dense components.

**Example:**
```python
def hybrid_reward(state, action, next_state, done, info):
    """Combine sparse goal reward with dense progress reward."""
    reward = 0.0
    
    # Dense component: progress toward goal
    old_distance = np.linalg.norm(state['position'] - info['goal_position'])
    new_distance = np.linalg.norm(next_state['position'] - info['goal_position'])
    progress = old_distance - new_distance
    reward += 0.1 * progress  # Small weight for shaping
    
    # Sparse component: goal achievement
    if done and info.get('success', False):
        reward += 10.0  # Large bonus for success
    
    return reward
```

### Reward Shaping

Reward shaping adds potential-based terms to guide learning without changing optimal policy:

```
R'(s, a, s') = R(s, a, s') + γΦ(s') - Φ(s)
```

Where:
- `Φ(s)` is a potential function
- `γ` is the discount factor
- This preserves optimal policy under certain conditions

## Designing Reward Functions

### Step 1: Define Success Criteria

Clearly specify what constitutes success:

```python
class RewardConfig:
    """Configuration for reward computation."""
    
    # Success criteria
    success_threshold: float = 0.95  # Accuracy threshold
    max_steps: int = 1000  # Maximum episode length
    goal_tolerance: float = 0.1  # Distance to goal
    
    # Reward components
    success_reward: float = 10.0
    step_penalty: float = -0.01
    failure_penalty: float = -5.0
```

### Step 2: Identify Desired Behaviors

List behaviors you want to encourage or discourage:

```python
def compute_behavioral_rewards(state, action, next_state):
    """Compute rewards for desired behaviors."""
    rewards = {}
    
    # Encourage movement toward goal
    rewards['progress'] = compute_progress_reward(state, next_state)
    
    # Discourage collisions
    rewards['collision'] = -1.0 if next_state['collision'] else 0.0
    
    # Encourage energy efficiency
    rewards['energy'] = -0.001 * np.sum(np.square(action))
    
    # Encourage smooth control
    rewards['smoothness'] = -0.01 * np.sum(np.abs(action - state['last_action']))
    
    return rewards
```

### Step 3: Balance Components

Assign relative weights to different reward components:

```python
class RewardWeights:
    """Weights for different reward components."""
    
    success: float = 10.0      # Main objective
    progress: float = 0.5       # Intermediate progress
    collision: float = 2.0      # Safety
    energy: float = 0.01        # Efficiency
    smoothness: float = 0.05    # Control quality
    time: float = 0.01          # Time penalty
```

### Step 4: Implement Reward Function

```python
def compute_reward(state, action, next_state, done, info, weights):
    """
    Compute total reward as weighted sum of components.
    
    Args:
        state: Current state
        action: Action taken
        next_state: Resulting state
        done: Episode termination flag
        info: Additional information
        weights: RewardWeights instance
        
    Returns:
        reward: Scalar reward value
        reward_info: Dictionary with component breakdown
    """
    reward_components = {}
    
    # Success/failure rewards (sparse)
    if done:
        if info.get('success', False):
            reward_components['success'] = weights.success
        else:
            reward_components['failure'] = -weights.success * 0.5
    else:
        reward_components['success'] = 0.0
        reward_components['failure'] = 0.0
    
    # Progress reward (dense)
    old_dist = np.linalg.norm(state['position'] - info['goal'])
    new_dist = np.linalg.norm(next_state['position'] - info['goal'])
    progress = old_dist - new_dist
    reward_components['progress'] = weights.progress * progress
    
    # Collision penalty
    if next_state.get('collision', False):
        reward_components['collision'] = -weights.collision
    else:
        reward_components['collision'] = 0.0
    
    # Energy efficiency
    energy_cost = np.sum(np.square(action))
    reward_components['energy'] = -weights.energy * energy_cost
    
    # Control smoothness
    if 'last_action' in state:
        action_diff = np.sum(np.abs(action - state['last_action']))
        reward_components['smoothness'] = -weights.smoothness * action_diff
    else:
        reward_components['smoothness'] = 0.0
    
    # Time penalty (encourage faster completion)
    reward_components['time'] = -weights.time
    
    # Compute total reward
    total_reward = sum(reward_components.values())
    
    return total_reward, reward_components
```

## Implementation Guide

### Basic Reward Function Class

```python
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

class BaseRewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward function.
        
        Args:
            config: Dictionary with reward configuration
        """
        self.config = config
        self.episode_rewards = []
        self.component_history = []
    
    @abstractmethod
    def compute(self, state, action, next_state, done, info):
        """
        Compute reward for a transition.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            done: Episode termination flag
            info: Additional information
            
        Returns:
            reward: Scalar reward value
            reward_info: Dictionary with component breakdown
        """
        pass
    
    def reset(self):
        """Reset episode-specific state."""
        self.episode_rewards = []
        self.component_history = []
    
    def log_reward(self, reward, components):
        """Log reward and components for analysis."""
        self.episode_rewards.append(reward)
        self.component_history.append(components)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about reward distribution."""
        if not self.episode_rewards:
            return {}
        
        rewards = np.array(self.episode_rewards)
        stats = {
            'total': np.sum(rewards),
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
        }
        
        # Component statistics
        if self.component_history:
            components = {}
            for key in self.component_history[0].keys():
                values = [c[key] for c in self.component_history]
                components[f'{key}_total'] = np.sum(values)
                components[f'{key}_mean'] = np.mean(values)
            stats.update(components)
        
        return stats
```

### Example Implementation: Navigation Task

```python
class NavigationReward(BaseRewardFunction):
    """Reward function for navigation tasks."""
    
    def __init__(self, config):
        super().__init__(config)
        self.goal_threshold = config.get('goal_threshold', 0.1)
        self.collision_penalty = config.get('collision_penalty', -1.0)
        self.success_reward = config.get('success_reward', 10.0)
        self.progress_weight = config.get('progress_weight', 1.0)
        self.step_penalty = config.get('step_penalty', -0.01)
        
        self.last_distance = None
    
    def compute(self, state, action, next_state, done, info):
        """Compute navigation reward."""
        components = {}
        
        # Calculate distance to goal
        goal = info['goal_position']
        current_pos = next_state['position']
        distance = np.linalg.norm(current_pos - goal)
        
        # Success reward
        if distance < self.goal_threshold:
            components['success'] = self.success_reward
        else:
            components['success'] = 0.0
        
        # Progress reward (distance decreased)
        if self.last_distance is not None:
            progress = self.last_distance - distance
            components['progress'] = self.progress_weight * progress
        else:
            components['progress'] = 0.0
        
        self.last_distance = distance
        
        # Collision penalty
        if next_state.get('collision', False):
            components['collision'] = self.collision_penalty
        else:
            components['collision'] = 0.0
        
        # Step penalty (encourage efficiency)
        components['step'] = self.step_penalty
        
        # Total reward
        total_reward = sum(components.values())
        
        # Log for analysis
        self.log_reward(total_reward, components)
        
        # Reset distance tracking if episode ends
        if done:
            self.last_distance = None
        
        return total_reward, components
```

## Tuning Strategies

### 1. Component Analysis

Track and visualize reward components separately:

```python
def analyze_reward_components(reward_function, episodes=100):
    """Analyze contribution of each reward component."""
    component_totals = {}
    
    for components in reward_function.component_history:
        for key, value in components.items():
            if key not in component_totals:
                component_totals[key] = []
            component_totals[key].append(value)
    
    # Print statistics
    for key, values in component_totals.items():
        print(f"{key}:")
        print(f"  Total: {np.sum(values):.2f}")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std: {np.std(values):.4f}")
        print(f"  Range: [{np.min(values):.2f}, {np.max(values):.2f}]")
```

### 2. Reward Scaling

Ensure reward components have appropriate magnitudes:

```python
class RewardNormalizer:
    """Normalize rewards to reasonable range."""
    
    def __init__(self, clip_range=(-10, 10), momentum=0.99):
        self.clip_range = clip_range
        self.momentum = momentum
        self.running_mean = 0.0
        self.running_std = 1.0
    
    def normalize(self, reward):
        """Normalize reward using running statistics."""
        # Update running statistics
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * reward
        squared_diff = (reward - self.running_mean) ** 2
        self.running_std = np.sqrt(
            self.momentum * self.running_std**2 + (1 - self.momentum) * squared_diff
        )
        
        # Normalize
        normalized = (reward - self.running_mean) / (self.running_std + 1e-8)
        
        # Clip to range
        clipped = np.clip(normalized, self.clip_range[0], self.clip_range[1])
        
        return clipped
```

### 3. Automated Tuning

Use grid search or optimization to find good weights:

```python
def tune_reward_weights(base_config, param_grid, num_trials=10):
    """
    Tune reward weights using grid search.
    
    Args:
        base_config: Base configuration
        param_grid: Dictionary of parameter ranges
        num_trials: Number of training runs per configuration
        
    Returns:
        best_config: Configuration with best performance
        results: DataFrame with all results
    """
    import itertools
    import pandas as pd
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    
    for config in configs:
        # Merge with base config
        full_config = {**base_config, **config}
        
        # Run multiple trials
        trial_results = []
        for trial in range(num_trials):
            # Train and evaluate
            performance = train_and_evaluate(full_config, seed=trial)
            trial_results.append(performance)
        
        # Record results
        results.append({
            'config': config,
            'mean_performance': np.mean(trial_results),
            'std_performance': np.std(trial_results),
        })
    
    # Find best configuration
    results_df = pd.DataFrame(results)
    best_idx = results_df['mean_performance'].idxmax()
    best_config = results_df.loc[best_idx, 'config']
    
    return best_config, results_df
```

### 4. Curriculum-Based Tuning

Gradually increase task difficulty and adjust rewards:

```python
class CurriculumReward(BaseRewardFunction):
    """Reward function that adapts based on curriculum stage."""
    
    def __init__(self, config):
        super().__init__(config)
        self.stage = 0
        self.stage_thresholds = config['stage_thresholds']
        self.stage_weights = config['stage_weights']
    
    def update_stage(self, performance):
        """Update curriculum stage based on performance."""
        for i, threshold in enumerate(self.stage_thresholds):
            if performance >= threshold:
                self.stage = i + 1
            else:
                break
    
    def compute(self, state, action, next_state, done, info):
        """Compute reward with stage-appropriate weights."""
        weights = self.stage_weights[self.stage]
        
        # Compute components with current weights
        components = self._compute_components(state, action, next_state, done, info)
        
        # Apply stage-specific weights
        weighted_components = {
            k: v * weights.get(k, 1.0) 
            for k, v in components.items()
        }
        
        total_reward = sum(weighted_components.values())
        
        return total_reward, weighted_components
```

## Common Patterns

### 1. Distance-Based Rewards

```python
def distance_reward(current_pos, goal_pos, last_distance=None, scale=1.0):
    """Reward based on distance to goal."""
    distance = np.linalg.norm(current_pos - goal_pos)
    
    if last_distance is not None:
        # Reward for decreasing distance
        progress = last_distance - distance
        return scale * progress, distance
    else:
        # Initial step, no reward
        return 0.0, distance
```

### 2. Time-Based Rewards

```python
def time_reward(step, max_steps, success=False):
    """Reward that encourages faster completion."""
    if success:
        # Bonus for early completion
        time_bonus = (max_steps - step) / max_steps
        return time_bonus
    else:
        # Small penalty per step
        return -0.01
```

### 3. Constraint-Based Penalties

```python
def constraint_penalty(state, constraints):
    """Penalty for violating constraints."""
    penalty = 0.0
    
    for constraint in constraints:
        if constraint['type'] == 'limit':
            # Penalize exceeding limits
            value = state[constraint['key']]
            if value > constraint['max']:
                penalty -= constraint['weight'] * (value - constraint['max'])
            elif value < constraint['min']:
                penalty -= constraint['weight'] * (constraint['min'] - value)
        
        elif constraint['type'] == 'forbidden':
            # Penalize forbidden states
            if state[constraint['key']] == constraint['value']:
                penalty -= constraint['weight']
    
    return penalty
```

### 4. Exploration Bonuses

```python
class ExplorationBonus:
    """Add bonus for visiting novel states."""
    
    def __init__(self, state_dim, bonus_scale=0.1):
        self.state_dim = state_dim
        self.bonus_scale = bonus_scale
        self.visit_counts = {}
    
    def compute_bonus(self, state):
        """Compute exploration bonus for state."""
        # Discretize state for counting
        state_key = tuple(np.round(state, decimals=1))
        
        # Get visit count
        count = self.visit_counts.get(state_key, 0)
        self.visit_counts[state_key] = count + 1
        
        # Bonus inversely proportional to visit count
        bonus = self.bonus_scale / np.sqrt(count + 1)
        
        return bonus
```

## Debugging Reward Functions

### Visualization Tools

```python
def visualize_reward_distribution(rewards, title="Reward Distribution"):
    """Visualize distribution of rewards."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Histogram
    plt.subplot(131)
    plt.hist(rewards, bins=50)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    
    # Time series
    plt.subplot(132)
    plt.plot(rewards)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward over Time')
    
    # Cumulative
    plt.subplot(133)
    plt.plot(np.cumsum(rewards))
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
```

### Sanity Checks

```python
def validate_reward_function(reward_fn, env, num_episodes=10):
    """Run sanity checks on reward function."""
    print("Running reward function validation...")
    
    all_rewards = []
    component_stats = {}
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = []
        done = False
        
        while not done:
            action = env.action_space.sample()
            next_state, _, done, info = env.step(action)
            
            # Compute reward
            reward, components = reward_fn.compute(state, action, next_state, done, info)
            episode_rewards.append(reward)
            
            # Track components
            for key, value in components.items():
                if key not in component_stats:
                    component_stats[key] = []
                component_stats[key].append(value)
            
            state = next_state
        
        all_rewards.extend(episode_rewards)
    
    # Check for issues
    issues = []
    
    # Check for NaN or Inf
    if np.any(np.isnan(all_rewards)) or np.any(np.isinf(all_rewards)):
        issues.append("WARNING: Found NaN or Inf rewards")
    
    # Check for extreme values
    if np.max(np.abs(all_rewards)) > 1000:
        issues.append(f"WARNING: Very large reward magnitude: {np.max(np.abs(all_rewards))}")
    
    # Check for zero variance
    if np.std(all_rewards) < 1e-6:
        issues.append("WARNING: Reward has near-zero variance")
    
    # Print results
    print(f"\nValidation Results ({num_episodes} episodes):")
    print(f"  Mean reward: {np.mean(all_rewards):.4f}")
    print(f"  Std reward: {np.std(all_rewards):.4f}")
    print(f"  Min reward: {np.min(all_rewards):.4f}")
    print(f"  Max reward: {np.max(all_rewards):.4f}")
    print(f"\nComponent contributions:")
    for key, values in component_stats.items():
        print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    if issues:
        print(f"\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo issues found!")
    
    return len(issues) == 0
```

## Best Practices

### 1. Start Simple

Begin with minimal reward function and add complexity gradually:

```python
# Phase 1: Sparse reward only
reward = 1.0 if success else 0.0

# Phase 2: Add dense progress signal
reward = 1.0 if success else -0.001 * distance_to_goal

# Phase 3: Add behavior shaping
reward = success_reward + progress_reward + efficiency_reward
```

### 2. Make Rewards Observable

Log and visualize all reward components:

```python
def log_reward_info(reward, components, step):
    """Log detailed reward information."""
    # Total reward
    wandb.log({'reward/total': reward}, step=step)
    
    # Individual components
    for key, value in components.items():
        wandb.log({f'reward/{key}': value}, step=step)
    
    # Component ratios
    total = sum(abs(v) for v in components.values())
    if total > 0:
        for key, value in components.items():
            ratio = abs(value) / total
            wandb.log({f'reward_ratio/{key}': ratio}, step=step)
```

### 3. Validate Against Human Intuition

Check if reward aligns with human judgment:

```python
def compare_trajectories(trajectory_a, trajectory_b, reward_fn):
    """Compare two trajectories and check if reward aligns with intuition."""
    reward_a = sum(reward_fn.compute(*step)[0] for step in trajectory_a)
    reward_b = sum(reward_fn.compute(*step)[0] for step in trajectory_b)
    
    print(f"Trajectory A: {reward_a:.2f}")
    print(f"Trajectory B: {reward_b:.2f}")
    print(f"Better trajectory: {'A' if reward_a > reward_b else 'B'}")
    
    # Manually verify if this matches expectation
```

### 4. Test Edge Cases

```python
def test_edge_cases(reward_fn, env):
    """Test reward function on edge cases."""
    test_cases = [
        ("Goal reached immediately", lambda: create_goal_state()),
        ("Collision occurs", lambda: create_collision_state()),
        ("Out of bounds", lambda: create_oob_state()),
        ("Maximum time exceeded", lambda: create_timeout_state()),
    ]
    
    for name, state_fn in test_cases:
        state = state_fn()
        reward, components = reward_fn.compute(state, None, state, True, {})
        print(f"{name}: {reward:.2f}")
        print(f"  Components: {components}")
```

### 5. Use Reward Clipping Carefully

```python
def clip_reward(reward, clip_range=(-10, 10)):
    """Clip reward to reasonable range."""
    clipped = np.clip(reward, clip_range[0], clip_range[1])
    
    if clipped != reward:
        # Log clipping events
        print(f"Warning: Reward {reward:.2f} clipped to {clipped:.2f}")
    
    return clipped
```

### 6. Version Control Reward Functions

Keep track of reward function changes:

```python
class RewardFunctionV1(BaseRewardFunction):
    """Initial version - sparse reward only."""
    VERSION = "1.0.0"
    # ...

class RewardFunctionV2(BaseRewardFunction):
    """Version 2 - added progress reward."""
    VERSION = "2.0.0"
    # ...

# Log version in experiments
config['reward_version'] = reward_fn.VERSION
```

### 7. Document Reward Design Decisions

```python
"""
Reward Function Design Notes
============================

Date: 2024-01-15
Version: 2.1.0

Design Rationale:
- Primary objective: Reach goal position (sparse reward: +10)
- Secondary objective: Minimize time (step penalty: -0.01)
- Constraint: Avoid collisions (penalty: -5)
- Shaping: Progress toward goal (coefficient: 0.5)

Tuning History:
- v1.0.0: Sparse reward only - too slow to learn
- v2.0.0: Added progress reward - learned faster but unstable
- v2.1.0: Reduced progress weight from 1.0 to 0.5 - improved stability

Known Issues:
- Occasionally learns to circle near goal without completing
- May require success threshold adjustment for different environments
"""
```

## Summary

Effective reward function design requires:

1. **Clear objectives**: Define what success means
2. **Appropriate density**: Balance sparse and dense rewards
3. **Proper scaling**: Ensure components have suitable magnitudes
4. **Systematic tuning**: Use data-driven approaches to find good weights
5. **Continuous monitoring**: Track and visualize reward components
6. **Iterative refinement**: Start simple and add complexity as needed
7. **Thorough validation**: Test edge cases and sanity check behavior

Remember: The reward function is your primary interface for communicating desired behavior to the agent. Invest time in getting it right!
