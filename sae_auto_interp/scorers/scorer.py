from dataclasses import dataclass
from ..features.features import FeatureRecord, Example
from abc import ABC, abstractmethod
from typing import List, Any
from typing import List

@dataclass
class ScorerInput():
    explanation: str
    record: FeatureRecord
    test_examples: List[Example]

@dataclass
class ScorerResult():
    input: Any
    response: str | List[str] = ""
    score: float = 0.0

class Scorer(ABC):
    def __init__(
        self,
        validate: bool = True,
    ):
        self.validate = validate
    
    @abstractmethod
    def __call__(
        self,
        scorer_in: ScorerInput,
    ) -> ScorerResult:
        pass