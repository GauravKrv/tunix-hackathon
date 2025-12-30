from typing import Dict, Any
from .base import RewardFunction


class ExplanationQualityReward(RewardFunction):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.min_length = self.config.get('min_length', 50)
        self.max_length = self.config.get('max_length', 1000)
        self.clarity_weight = self.config.get('clarity_weight', 0.4)
        self.completeness_weight = self.config.get('completeness_weight', 0.3)
        self.structure_weight = self.config.get('structure_weight', 0.3)
    
    def compute(self, explanation: str, **kwargs) -> float:
        if not explanation or not explanation.strip():
            return 0.0
        
        explanation = explanation.strip()
        
        clarity_score = self._score_clarity(explanation)
        completeness_score = self._score_completeness(explanation)
        structure_score = self._score_structure(explanation)
        
        quality_score = (
            self.clarity_weight * clarity_score +
            self.completeness_weight * completeness_score +
            self.structure_weight * structure_score
        )
        
        return self.normalize(quality_score, 0.0, 1.0)
    
    def _score_clarity(self, explanation: str) -> float:
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
    
    def _score_completeness(self, explanation: str) -> float:
        length = len(explanation)
        
        if length < self.min_length:
            length_score = length / self.min_length
        elif length > self.max_length:
            length_score = max(0.5, 1.0 - (length - self.max_length) / self.max_length)
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
    
    def _score_structure(self, explanation: str) -> float:
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
