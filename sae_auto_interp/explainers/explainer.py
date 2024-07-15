from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from ..features.features import Example, FeatureRecord


@dataclass
class ExplainerInput:
    train_examples: List[Example]
    record: FeatureRecord 


class Explainer(ABC):

    @abstractmethod
    def __call__(
        self,
        explainer_in: ExplainerInput
    ) -> str:
        pass