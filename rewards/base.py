from abc import ABC, abstractmethod
from typing import Any, Dict


class RewardFunction(ABC):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @abstractmethod
    def compute(self, **kwargs) -> float:
        pass
    
    def normalize(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        if score < min_val:
            return 0.0
        if score > max_val:
            return 1.0
        if max_val == min_val:
            return 0.0
        return (score - min_val) / (max_val - min_val)
