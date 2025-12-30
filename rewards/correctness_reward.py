from typing import Any, Dict
from .base import RewardFunction


class CorrectnessReward(RewardFunction):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.exact_match_weight = self.config.get('exact_match_weight', 0.6)
        self.partial_match_weight = self.config.get('partial_match_weight', 0.4)
    
    def compute(self, prediction: Any, ground_truth: Any, **kwargs) -> float:
        if prediction is None or ground_truth is None:
            return 0.0
        
        pred_str = str(prediction).strip().lower()
        truth_str = str(ground_truth).strip().lower()
        
        if pred_str == truth_str:
            return 1.0
        
        if not pred_str or not truth_str:
            return 0.0
        
        exact_match_score = 1.0 if pred_str == truth_str else 0.0
        
        pred_tokens = set(pred_str.split())
        truth_tokens = set(truth_str.split())
        
        if not truth_tokens:
            return 0.0
        
        intersection = pred_tokens & truth_tokens
        union = pred_tokens | truth_tokens
        
        if not union:
            return 0.0
        
        jaccard_similarity = len(intersection) / len(union)
        partial_match_score = jaccard_similarity
        
        final_score = (
            self.exact_match_weight * exact_match_score +
            self.partial_match_weight * partial_match_score
        )
        
        return self.normalize(final_score, 0.0, 1.0)
