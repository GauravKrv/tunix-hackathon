def conciseness_reward(
    explanation: str,
    min_length: int = 50,
    max_length: int = 1000,
    clarity_weight: float = 0.4,
    completeness_weight: float = 0.3,
    structure_weight: float = 0.3,
    **kwargs
) -> float:
    """
    Compute conciseness reward based on explanation quality.
    
    Args:
        explanation: The explanation text to evaluate
        min_length: Minimum expected length (default: 50)
        max_length: Maximum expected length (default: 1000)
        clarity_weight: Weight for clarity score (default: 0.4)
        completeness_weight: Weight for completeness score (default: 0.3)
        structure_weight: Weight for structure score (default: 0.3)
        **kwargs: Additional keyword arguments (ignored)
    
    Returns:
        Reward score between 0.0 and 1.0
    """
    if not explanation or not explanation.strip():
        return 0.0
    
    explanation = explanation.strip()
    
    clarity_score = _score_clarity(explanation)
    completeness_score = _score_completeness(explanation, min_length, max_length)
    structure_score = _score_structure(explanation)
    
    quality_score = (
        clarity_weight * clarity_score +
        completeness_weight * completeness_score +
        structure_weight * structure_score
    )
    
    return max(0.0, min(1.0, quality_score))


def _score_clarity(explanation: str) -> float:
    """Score based on clarity of explanation."""
    words = explanation.split()
    num_words = len(words)
    
    if num_words == 0:
        return 0.0
    
    avg_word_length = sum(len(word) for word in words) / num_words
    
    sentences = [s.strip() for s in explanation.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    num_sentences = len(sentences)
    
    if num_sentences == 0:
        return 0.0
    
    avg_sentence_length = num_words / num_sentences
    
    ideal_word_length = 5.0
    word_length_score = 1.0 - min(1.0, abs(avg_word_length - ideal_word_length) / ideal_word_length)
    
    ideal_sentence_length = 15.0
    sentence_length_score = 1.0 - min(1.0, abs(avg_sentence_length - ideal_sentence_length) / ideal_sentence_length)
    
    clarity_indicators = ['clearly', 'specifically', 'namely', 'for example', 'such as', 'in other words']
    has_clarity_indicators = any(indicator in explanation.lower() for indicator in clarity_indicators)
    indicator_score = 1.0 if has_clarity_indicators else 0.7
    
    return (word_length_score + sentence_length_score + indicator_score) / 3


def _score_completeness(explanation: str, min_length: int, max_length: int) -> float:
    """Score based on completeness of explanation."""
    length = len(explanation)
    
    if length < min_length:
        length_score = length / min_length
    elif length > max_length:
        length_score = max(0.5, 1.0 - (length - max_length) / max_length)
    else:
        length_score = 1.0
    
    completeness_indicators = [
        'because', 'therefore', 'due to', 'as a result', 'this means',
        'in conclusion', 'overall', 'in summary', 'consequently'
    ]
    indicator_count = sum(1 for indicator in completeness_indicators if indicator in explanation.lower())
    indicator_score = min(1.0, indicator_count / 3)
    
    has_multiple_sentences = len([s for s in explanation.split('.') if s.strip()]) > 1
    sentence_score = 1.0 if has_multiple_sentences else 0.6
    
    return (length_score + indicator_score + sentence_score) / 3


def _score_structure(explanation: str) -> float:
    """Score based on structure of explanation."""
    has_paragraphs = '\n\n' in explanation or '\n' in explanation
    paragraph_score = 1.0 if has_paragraphs else 0.7
    
    sentences = [s.strip() for s in explanation.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    
    if not sentences:
        return 0.0
    
    capitalized_sentences = sum(1 for s in sentences if s and s[0].isupper())
    capitalization_score = capitalized_sentences / len(sentences)
    
    structure_markers = ['first', 'second', 'third', 'finally', 'next', 'then', 'lastly']
    has_structure_markers = any(marker in explanation.lower() for marker in structure_markers)
    marker_score = 1.0 if has_structure_markers else 0.8
    
    punctuation_count = sum(1 for c in explanation if c in '.!?,;:')
    word_count = len(explanation.split())
    punctuation_ratio = punctuation_count / word_count if word_count > 0 else 0
    punctuation_score = min(1.0, punctuation_ratio * 10)
    
    return (paragraph_score + capitalization_score + marker_score + punctuation_score) / 4
