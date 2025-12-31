from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class RewardComponent:
    """A single reward function component with its configuration."""
    name: str
    reward_fn: Callable[..., float]
    weight: float = 1.0
    normalize: bool = False
    clip_range: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute(self, *args, **kwargs) -> float:
        """Compute the reward value with optional normalization and clipping."""
        reward = self.reward_fn(*args, **kwargs)
        
        if self.clip_range is not None:
            reward = np.clip(reward, self.clip_range[0], self.clip_range[1])
        
        if self.normalize:
            if self.clip_range is not None:
                min_val, max_val = self.clip_range
                if max_val > min_val:
                    reward = (reward - min_val) / (max_val - min_val)
        
        return reward


@dataclass
class ProcessStep:
    """Represents a single step in a reasoning process."""
    step_index: int
    content: str
    action: Optional[str] = None
    observation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessRewardResult:
    """Result of process reward modeling evaluation."""
    step_scores: List[float]
    aggregate_score: float
    step_details: List[Dict[str, Any]]
    trajectory_score: Optional[float] = None


def compute_composite_reward(
    components: List[RewardComponent],
    *args,
    **kwargs
) -> float:
    """
    Compute composite reward using weighted summation.
    
    Args:
        components: List of reward components to combine
        *args: Positional arguments passed to each component
        **kwargs: Keyword arguments passed to each component
    
    Returns:
        Weighted sum of component rewards
    """
    if not components:
        return 0.0
    
    total_reward = 0.0
    for component in components:
        score = component.compute(*args, **kwargs)
        total_reward += component.weight * score
    
    return total_reward


def compute_composite_reward_with_details(
    components: List[RewardComponent],
    *args,
    **kwargs
) -> Tuple[float, Dict[str, float]]:
    """
    Compute composite reward with individual component scores.
    
    Args:
        components: List of reward components to combine
        *args: Positional arguments passed to each component
        **kwargs: Keyword arguments passed to each component
    
    Returns:
        Tuple of (composite_score, component_scores_dict)
    """
    if not components:
        return 0.0, {}
    
    component_scores = {}
    total_reward = 0.0
    
    for component in components:
        score = component.compute(*args, **kwargs)
        component_scores[component.name] = score
        total_reward += component.weight * score
    
    return total_reward, component_scores


def score_reasoning_step(
    step: ProcessStep,
    step_reward_components: List[RewardComponent],
    context: Optional[Dict[str, Any]] = None
) -> float:
    """
    Score a single reasoning step using configured reward components.
    
    Args:
        step: The reasoning step to score
        step_reward_components: Reward components for scoring
        context: Optional context information
    
    Returns:
        Weighted sum of step rewards
    """
    context = context or {}
    return compute_composite_reward(
        step_reward_components,
        step=step,
        context=context
    )


def score_reasoning_trajectory(
    steps: List[ProcessStep],
    step_reward_components: List[RewardComponent],
    trajectory_reward_components: Optional[List[RewardComponent]] = None,
    discount_factor: float = 1.0,
    context: Optional[Dict[str, Any]] = None
) -> ProcessRewardResult:
    """
    Score an entire reasoning trajectory.
    
    Args:
        steps: List of reasoning steps
        step_reward_components: Reward components for scoring individual steps
        trajectory_reward_components: Optional reward components for scoring entire trajectory
        discount_factor: Discount factor for future steps (gamma in RL)
        context: Optional context information
    
    Returns:
        ProcessRewardResult containing step scores and aggregate score
    """
    context = context or {}
    
    step_scores = []
    step_details = []
    
    for i, step in enumerate(steps):
        step_context = {**context, 'step_index': i, 'total_steps': len(steps)}
        score = score_reasoning_step(step, step_reward_components, step_context)
        
        discounted_score = score * (discount_factor ** i)
        step_scores.append(discounted_score)
        
        step_details.append({
            'step_index': i,
            'raw_score': score,
            'discounted_score': discounted_score,
            'content': step.content,
            'action': step.action
        })
    
    aggregate_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
    
    trajectory_score = None
    if trajectory_reward_components:
        trajectory_score = compute_composite_reward(
            trajectory_reward_components,
            steps=steps,
            context=context
        )
    
    return ProcessRewardResult(
        step_scores=step_scores,
        aggregate_score=aggregate_score,
        step_details=step_details,
        trajectory_score=trajectory_score
    )


def create_reward_components(
    reward_functions: Dict[str, Callable],
    weights: Optional[Dict[str, float]] = None
) -> List[RewardComponent]:
    """
    Helper function to create reward components from functions and weights.
    
    Args:
        reward_functions: Dictionary mapping names to reward functions
        weights: Optional dictionary mapping names to weights (default: 1.0 for all)
    
    Returns:
        List of RewardComponent objects
    """
    weights = weights or {}
    components = [
        RewardComponent(
            name=name,
            reward_fn=fn,
            weight=weights.get(name, 1.0)
        )
        for name, fn in reward_functions.items()
    ]
    return components
