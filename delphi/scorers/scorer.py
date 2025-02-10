from abc import ABC, abstractmethod
from typing import Any, NamedTuple

from ..latents.latents import LatentRecord


class ScorerResult(NamedTuple):
    record: LatentRecord
    """Latent record passed through."""

    score: Any
    """Generated score for latent."""


class Scorer(ABC):
    @abstractmethod
    def __call__(self, record: LatentRecord) -> ScorerResult:
        pass
