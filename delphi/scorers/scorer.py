from abc import ABC, abstractmethod
from typing import Any, NamedTuple

from ..features.features import FeatureRecord


class ScorerResult(NamedTuple):
    record: FeatureRecord
    """Feature record passed through."""

    score: Any
    """Generated score for feature."""


class Scorer(ABC):
    @abstractmethod
    def __call__(self, record: FeatureRecord) -> ScorerResult:
        pass
