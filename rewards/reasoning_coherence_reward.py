from typing import Dict, Any, List
from .base import RewardFunction


class ReasoningCoherenceReward(RewardFunction):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.min_steps = self.config.get('min_steps', 1)
        self.max_steps = self.config.get('max_steps', 10)
        self.logical_flow_weight = self.config.get('logical_flow_weight', 0.5)
        self.step_quality_weight = self.config.get('step_quality_weight', 0.5)
    
    def compute(self, reasoning_steps: List[str], **kwargs) -> float:
        if not reasoning_steps:
            return 0.0
        
        num_steps = len(reasoning_steps)
        
        step_count_score = self._score_step_count(num_steps)
        logical_flow_score = self._score_logical_flow(reasoning_steps)
        step_quality_score = self._score_step_quality(reasoning_steps)
        
        coherence_score = (
            self.logical_flow_weight * logical_flow_score +
            self.step_quality_weight * step_quality_score
        ) * step_count_score
        
        return self.normalize(coherence_score, 0.0, 1.0)
    
    def _score_step_count(self, num_steps: int) -> float:
        if num_steps < self.min_steps:
            return num_steps / self.min_steps
        if num_steps > self.max_steps:
            return max(0.0, 1.0 - (num_steps - self.max_steps) * 0.05)
        return 1.0
    
    def _score_logical_flow(self, reasoning_steps: List[str]) -> float:
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
    
    def _score_step_quality(self, reasoning_steps: List[str]) -> float:
        if not reasoning_steps:
            return 0.0
        
        min_step_length = self.config.get('min_step_length', 10)
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
