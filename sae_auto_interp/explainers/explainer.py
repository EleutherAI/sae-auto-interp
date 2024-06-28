from dataclasses import dataclass

from ..autoencoders.features import Example, FeatureRecord  
from typing import List
import time


from abc import ABC, abstractmethod

@dataclass
class ExplainerInput:
    train_examples: List[Example]
    record: FeatureRecord 

@dataclass
class ExplainerResult:
    explainer_type: str = ""
    input: str = ""
    response: str = ""
    explanation: str = ""

class Explainer(ABC):

    @abstractmethod
    def __call__(
        self,
        explainer_in: ExplainerInput
    ) -> ExplainerResult:
        pass
        
def run_explainers(
    explainers: List[Explainer],
    explainer_in: ExplainerInput,
    logging = None
):
    logger = logging.info if logging else print
    for explainer in explainers:
        
        def run():
            name = explainer.name

            logger(f"Running explainer {name}")

            start = time.time()
            result = explainer(explainer_in)
            end = time.time()

            logger(f"Finished explainer {name}")

            runtime = end - start

            return runtime, result 

        yield explainer.name, run
            
