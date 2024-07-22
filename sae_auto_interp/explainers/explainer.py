from abc import ABC, abstractmethod
from typing import NamedTuple

from ..features.features import FeatureRecord

class ExplainerResult(NamedTuple):
    record: FeatureRecord
    """Feature record passed through to scorer."""

    explanation: str
    """Generated explanation for feature."""


class Explainer(ABC):

    @abstractmethod
    def __call__(
        self,
        record: FeatureRecord
    ):
        pass


def explanation_loader(record: FeatureRecord, explanation_dir: str) -> str:

    with open(f'{explanation_dir}/{record.id}.txt', 'r') as f:
        explanation = f.read()
    
    return ExplainerResult(
        record=record,
        explanation=explanation
    )

