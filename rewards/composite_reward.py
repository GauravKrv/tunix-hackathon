from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class CompositionStrategy(Enum):
    """Strategies for combining multiple reward functions."""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    WEIGHTED_GEOMETRIC_MEAN = "weighted_geometric_mean"
    MIN = "min"
    MAX = "max"


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
            # Normalize to [0, 1] range if clip_range is provided
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


class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        """Compute the reward."""
        pass


class ProcessRewardModel(ABC):
    """Abstract base class for process reward models."""
    
    @abstractmethod
    def score_step(self, step: ProcessStep, context: Optional[Dict[str, Any]] = None) -> float:
        """Score a single reasoning step."""
        pass
    
    @abstractmethod
    def score_trajectory(self, steps: List[ProcessStep], context: Optional[Dict[str, Any]] = None) -> ProcessRewardResult:
        """Score an entire reasoning trajectory."""
        pass


class SimpleProcessRewardModel(ProcessRewardModel):
    """Simple implementation of process reward modeling using reward components."""
    
    def __init__(
        self,
        step_reward_components: List[RewardComponent],
        trajectory_reward_components: Optional[List[RewardComponent]] = None,
        composition_strategy: CompositionStrategy = CompositionStrategy.ADDITIVE,
        discount_factor: float = 1.0,
        early_termination_penalty: float = 0.0
    ):
        """
        Initialize the process reward model.
        
        Args:
            step_reward_components: Reward components for scoring individual steps
            trajectory_reward_components: Optional reward components for scoring entire trajectories
            composition_strategy: How to combine multiple reward components
            discount_factor: Discount factor for future steps (gamma in RL)
            early_termination_penalty: Penalty applied if reasoning terminates early
        """
        self.step_reward_components = step_reward_components
        self.trajectory_reward_components = trajectory_reward_components or []
        self.composition_strategy = composition_strategy
        self.discount_factor = discount_factor
        self.early_termination_penalty = early_termination_penalty
    
    def score_step(self, step: ProcessStep, context: Optional[Dict[str, Any]] = None) -> float:
        """Score a single reasoning step using configured reward components."""
        context = context or {}
        
        rewards = []
        weights = []
        
        for component in self.step_reward_components:
            reward = component.compute(step=step, context=context)
            rewards.append(reward)
            weights.append(component.weight)
        
        return self._combine_rewards(rewards, weights)
    
    def score_trajectory(self, steps: List[ProcessStep], context: Optional[Dict[str, Any]] = None) -> ProcessRewardResult:
        """Score an entire reasoning trajectory."""
        context = context or {}
        
        step_scores = []
        step_details = []
        
        # Score each individual step
        for i, step in enumerate(steps):
            step_context = {**context, 'step_index': i, 'total_steps': len(steps)}
            score = self.score_step(step, step_context)
            
            # Apply discount factor
            discounted_score = score * (self.discount_factor ** i)
            step_scores.append(discounted_score)
            
            step_details.append({
                'step_index': i,
                'raw_score': score,
                'discounted_score': discounted_score,
                'content': step.content,
                'action': step.action
            })
        
        # Compute aggregate score from step scores
        aggregate_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
        
        # Optionally score the entire trajectory
        trajectory_score = None
        if self.trajectory_reward_components:
            traj_rewards = []
            traj_weights = []
            
            for component in self.trajectory_reward_components:
                reward = component.compute(steps=steps, context=context)
                traj_rewards.append(reward)
                traj_weights.append(component.weight)
            
            trajectory_score = self._combine_rewards(traj_rewards, traj_weights)
        
        return ProcessRewardResult(
            step_scores=step_scores,
            aggregate_score=aggregate_score,
            step_details=step_details,
            trajectory_score=trajectory_score
        )
    
    def _combine_rewards(self, rewards: List[float], weights: List[float]) -> float:
        """Combine multiple rewards according to the composition strategy."""
        if not rewards:
            return 0.0
        
        rewards = np.array(rewards)
        weights = np.array(weights)
        
        if self.composition_strategy == CompositionStrategy.ADDITIVE:
            return np.sum(rewards * weights)
        
        elif self.composition_strategy == CompositionStrategy.MULTIPLICATIVE:
            # Apply weights as exponents
            result = 1.0
            for r, w in zip(rewards, weights):
                result *= (r ** w)
            return result
        
        elif self.composition_strategy == CompositionStrategy.WEIGHTED_GEOMETRIC_MEAN:
            # Weighted geometric mean
            total_weight = np.sum(weights)
            if total_weight == 0:
                return 0.0
            normalized_weights = weights / total_weight
            result = np.prod(np.power(rewards, normalized_weights))
            return result
        
        elif self.composition_strategy == CompositionStrategy.MIN:
            return np.min(rewards)
        
        elif self.composition_strategy == CompositionStrategy.MAX:
            return np.max(rewards)
        
        else:
            raise ValueError(f"Unknown composition strategy: {self.composition_strategy}")


class CompositeReward:
    """
    Combines multiple reward functions with configurable weights and composition strategies.
    Supports both additive and multiplicative composition as well as advanced strategies.
    """
    
    def __init__(
        self,
        components: List[RewardComponent],
        strategy: CompositionStrategy = CompositionStrategy.ADDITIVE,
        normalize_weights: bool = False,
        return_components: bool = False
    ):
        """
        Initialize the composite reward.
        
        Args:
            components: List of reward components to combine
            strategy: Strategy for combining rewards
            normalize_weights: Whether to normalize weights to sum to 1
            return_components: Whether to return individual component scores
        """
        self.components = components
        self.strategy = strategy
        self.normalize_weights = normalize_weights
        self.return_components = return_components
        
        if normalize_weights:
            self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize component weights to sum to 1."""
        total_weight = sum(comp.weight for comp in self.components)
        if total_weight > 0:
            for comp in self.components:
                comp.weight = comp.weight / total_weight
    
    def __call__(self, *args, **kwargs) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Compute the composite reward.
        
        Returns:
            If return_components is False: composite reward score
            If return_components is True: tuple of (composite_score, component_scores_dict)
        """
        component_scores = {}
        rewards = []
        weights = []
        
        for component in self.components:
            score = component.compute(*args, **kwargs)
            component_scores[component.name] = score
            rewards.append(score)
            weights.append(component.weight)
        
        composite_score = self._combine_rewards(rewards, weights)
        
        if self.return_components:
            return composite_score, component_scores
        return composite_score
    
    def _combine_rewards(self, rewards: List[float], weights: List[float]) -> float:
        """Combine multiple rewards according to the composition strategy."""
        if not rewards:
            return 0.0
        
        rewards = np.array(rewards)
        weights = np.array(weights)
        
        if self.strategy == CompositionStrategy.ADDITIVE:
            return float(np.sum(rewards * weights))
        
        elif self.strategy == CompositionStrategy.MULTIPLICATIVE:
            result = 1.0
            for r, w in zip(rewards, weights):
                result *= (r ** w)
            return float(result)
        
        elif self.strategy == CompositionStrategy.WEIGHTED_GEOMETRIC_MEAN:
            total_weight = np.sum(weights)
            if total_weight == 0:
                return 0.0
            normalized_weights = weights / total_weight
            result = np.prod(np.power(np.abs(rewards), normalized_weights))
            # Preserve sign based on weighted average of signs
            sign = np.sign(np.sum(rewards * normalized_weights))
            return float(sign * result)
        
        elif self.strategy == CompositionStrategy.MIN:
            return float(np.min(rewards))
        
        elif self.strategy == CompositionStrategy.MAX:
            return float(np.max(rewards))
        
        else:
            raise ValueError(f"Unknown composition strategy: {self.strategy}")
    
    def add_component(self, component: RewardComponent):
        """Add a new reward component dynamically."""
        self.components.append(component)
        if self.normalize_weights:
            self._normalize_weights()
    
    def remove_component(self, name: str) -> bool:
        """Remove a reward component by name. Returns True if found and removed."""
        for i, comp in enumerate(self.components):
            if comp.name == name:
                self.components.pop(i)
                if self.normalize_weights:
                    self._normalize_weights()
                return True
        return False
    
    def update_weight(self, name: str, weight: float) -> bool:
        """Update the weight of a specific component. Returns True if found and updated."""
        for comp in self.components:
            if comp.name == name:
                comp.weight = weight
                if self.normalize_weights:
                    self._normalize_weights()
                return True
        return False
    
    def get_component(self, name: str) -> Optional[RewardComponent]:
        """Get a reward component by name."""
        for comp in self.components:
            if comp.name == name:
                return comp
        return None
    
    def evaluate_components(self, *args, **kwargs) -> Dict[str, float]:
        """Evaluate all components individually and return their scores."""
        return {comp.name: comp.compute(*args, **kwargs) for comp in self.components}


class AdaptiveCompositeReward(CompositeReward):
    """
    Composite reward with adaptive weight adjustment based on performance history.
    """
    
    def __init__(
        self,
        components: List[RewardComponent],
        strategy: CompositionStrategy = CompositionStrategy.ADDITIVE,
        normalize_weights: bool = True,
        return_components: bool = False,
        adaptation_rate: float = 0.1,
        history_size: int = 100
    ):
        """
        Initialize adaptive composite reward.
        
        Args:
            components: List of reward components
            strategy: Composition strategy
            normalize_weights: Whether to normalize weights
            return_components: Whether to return individual components
            adaptation_rate: Rate at which weights adapt (0 to 1)
            history_size: Number of recent evaluations to track
        """
        super().__init__(components, strategy, normalize_weights, return_components)
        self.adaptation_rate = adaptation_rate
        self.history_size = history_size
        self.component_history = {comp.name: [] for comp in components}
    
    def __call__(self, *args, **kwargs) -> Union[float, Tuple[float, Dict[str, float]]]:
        """Compute reward and update history for adaptation."""
        result = super().__call__(*args, **kwargs)
        
        # Extract component scores
        if self.return_components:
            composite_score, component_scores = result
        else:
            composite_score = result
            component_scores = self.evaluate_components(*args, **kwargs)
        
        # Update history
        for name, score in component_scores.items():
            if name in self.component_history:
                self.component_history[name].append(score)
                if len(self.component_history[name]) > self.history_size:
                    self.component_history[name].pop(0)
        
        return result
    
    def adapt_weights(self, target_balance: Optional[Dict[str, float]] = None):
        """
        Adapt component weights based on their performance variance.
        Components with higher variance get slightly reduced weight for stability.
        
        Args:
            target_balance: Optional target weight distribution
        """
        if target_balance is None:
            # Adapt based on inverse variance (reward stable components)
            variances = {}
            for comp in self.components:
                history = self.component_history[comp.name]
                if len(history) > 1:
                    variances[comp.name] = np.var(history)
                else:
                    variances[comp.name] = 1.0
            
            # Compute inverse variance weights
            inv_var_weights = {name: 1.0 / (var + 1e-8) for name, var in variances.items()}
            total = sum(inv_var_weights.values())
            target_balance = {name: w / total for name, w in inv_var_weights.items()}
        
        # Gradually adjust weights towards target
        for comp in self.components:
            if comp.name in target_balance:
                target_weight = target_balance[comp.name]
                comp.weight = (1 - self.adaptation_rate) * comp.weight + \
                              self.adaptation_rate * target_weight
        
        if self.normalize_weights:
            self._normalize_weights()


def create_simple_additive_reward(
    reward_functions: Dict[str, Callable],
    weights: Optional[Dict[str, float]] = None
) -> CompositeReward:
    """
    Helper function to create a simple additive composite reward.
    
    Args:
        reward_functions: Dictionary mapping names to reward functions
        weights: Optional dictionary mapping names to weights (default: 1.0 for all)
    
    Returns:
        CompositeReward configured for additive composition
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
    return CompositeReward(components, strategy=CompositionStrategy.ADDITIVE)


def create_simple_multiplicative_reward(
    reward_functions: Dict[str, Callable],
    weights: Optional[Dict[str, float]] = None
) -> CompositeReward:
    """
    Helper function to create a simple multiplicative composite reward.
    
    Args:
        reward_functions: Dictionary mapping names to reward functions
        weights: Optional dictionary mapping names to weights (used as exponents)
    
    Returns:
        CompositeReward configured for multiplicative composition
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
    return CompositeReward(components, strategy=CompositionStrategy.MULTIPLICATIVE)
