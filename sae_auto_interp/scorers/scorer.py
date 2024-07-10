from dataclasses import dataclass
from ..features.features import FeatureRecord, Example
from abc import ABC, abstractmethod
from typing import List, Any
from typing import List

@dataclass
class ScorerInput:
    explanation: str
    record: FeatureRecord
    test_examples: List[Example]
    random_examples: List[Example] = None
    extra_examples: List[Example] = None

class Scorer(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(
        self,
        scorer_in: ScorerInput,
    ) -> Any:
        pass