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


class Explainer(ABC):

    @abstractmethod
    def __call__(
        self,
        explainer_in: ExplainerInput
    ) -> str:
        pass