from typing import List


def structure_reward(
    reasoning_steps: List[str],
    min_steps: int = 1,
    max_steps: int = 10,
    logical_flow_weight: float = 0.5,
    step_quality_weight: float = 0.5,
    min_step_length: int = 10,
    **kwargs
) -> float:
    """
    Compute structure reward based on reasoning coherence and step quality.
    
    Args:
        reasoning_steps: List of reasoning step strings
        min_steps: Minimum number of expected steps (default: 1)
        max_steps: Maximum number of expected steps (default: 10)
        logical_flow_weight: Weight for logical flow score (default: 0.5)
        step_quality_weight: Weight for step quality score (default: 0.5)
        min_step_length: Minimum expected length of each step (default: 10)
        **kwargs: Additional keyword arguments (ignored)
    
    Returns:
        Reward score between 0.0 and 1.0
    """
    if not reasoning_steps:
        return 0.0
    
    num_steps = len(reasoning_steps)
    
    step_count_score = _score_step_count(num_steps, min_steps, max_steps)
    logical_flow_score = _score_logical_flow(reasoning_steps)
    step_quality_score = _score_step_quality(reasoning_steps, min_step_length)
    
    coherence_score = (
        logical_flow_weight * logical_flow_score +
        step_quality_weight * step_quality_score
    ) * step_count_score
    
    return max(0.0, min(1.0, coherence_score))


def _score_step_count(num_steps: int, min_steps: int, max_steps: int) -> float:
    """Score based on number of steps."""
    if num_steps < min_steps:
        return num_steps / min_steps
    if num_steps > max_steps:
        return max(0.0, 1.0 - (num_steps - max_steps) * 0.05)
    return 1.0


def _score_logical_flow(reasoning_steps: List[str]) -> float:
    """Score based on logical flow indicators."""
    if len(reasoning_steps) <= 1:
        return 1.0
    
    transition_words = [
        'therefore', 'thus', 'hence', 'consequently', 'because',
        'since', 'so', 'then', 'next', 'first', 'second', 'finally',
        'however', 'but', 'although', 'while', 'given', 'if', 'when'
    ]
    
    transitions_found = 0
    for step in reasoning_steps:
        step_lower = step.lower()
        if any(word in step_lower for word in transition_words):
            transitions_found += 1
    
    max_possible_transitions = len(reasoning_steps)
    flow_score = transitions_found / max_possible_transitions if max_possible_transitions > 0 else 0.0
    
    return min(1.0, flow_score)


def _score_step_quality(reasoning_steps: List[str], min_step_length: int) -> float:
    """Score based on quality of individual steps."""
    if not reasoning_steps:
        return 0.0
    
    quality_scores = []
    
    for step in reasoning_steps:
        step = step.strip()
        if not step:
            quality_scores.append(0.0)
            continue
        
        length_score = min(1.0, len(step) / min_step_length)
        
        has_punctuation = any(p in step for p in ['.', '!', '?', ',', ';'])
        punctuation_score = 1.0 if has_punctuation else 0.5
        
        step_score = (length_score + punctuation_score) / 2
        quality_scores.append(step_score)
    
    return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
