from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from ..features.features import Example, FeatureRecord


@dataclass
class ExplainerInput:
    """
    Input to the explainer. Contains the training 
    examples and the FeatureRecord to explain.

    Args:
        train_examples (List[Example]): List of training examples.
        record (FeatureRecord): FeatureRecord to explain.
    """
    train_examples: List[Example]
    record: FeatureRecord 

@dataclass
class ExplainerResult:
    """
    Result of the explainer. Contains the explainer type, 
    prompt, response, and explanation.

    Args:
        explainer_type (str): Type of explainer.
        prompt (str): Prompt for the explainer.
        response (str): Response from the explainer.
        explanation (str): Explanation from the explainer.
    """
    explainer_type: str = ""
    prompt: str = ""
    response: str = ""
    explanation: str = ""

class Explainer(ABC):

    @abstractmethod
    def __call__(
        self,
        explainer_in: ExplainerInput
    ) -> ExplainerResult:
        pass