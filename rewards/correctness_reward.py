from typing import Any


def correctness_reward(
    prediction: Any,
    ground_truth: Any,
    exact_match_weight: float = 0.6,
    partial_match_weight: float = 0.4,
    **kwargs
) -> float:
    """
    Compute correctness reward based on prediction and ground truth.
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        exact_match_weight: Weight for exact match score (default: 0.6)
        partial_match_weight: Weight for partial match score (default: 0.4)
        **kwargs: Additional keyword arguments (ignored)
    
    Returns:
        Reward score between 0.0 and 1.0
    """
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
        exact_match_weight * exact_match_score +
        partial_match_weight * partial_match_score
    )
    
    return max(0.0, min(1.0, final_score))
