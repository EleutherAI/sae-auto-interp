from dataclasses import dataclass
from ..features import FeatureRecord, Example
from abc import ABC, abstractmethod
from typing import List, Any
import time

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
    
def run_scorers(
    scorers: List[Scorer],
    scorer_in: ScorerInput,
    logging = None
):
    logger = logging.info if logging else print
    for scorer in scorers:

        def run():
            name = scorer.name

            logger(f"Running scorer {name}")

            start = time.time()
            result = scorer(scorer_in)
            end = time.time()

            logger(f"Finished scorer {name}")

            runtime = end - start

            return runtime, result
        
        yield scorer.name, run

