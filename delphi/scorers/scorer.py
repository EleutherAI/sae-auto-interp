from abc import ABC, abstractmethod
from typing import Any, NamedTuple

from ..latents.latents import LatentRecord


class ScorerResult(NamedTuple):
    record: LatentRecord
    """Latent record passed through."""

    score: Any
    """Generated score for latent."""

    def to_dict(self):
        """Convert the scorer result to a dictionary for serialization."""
        return {
            **asdict(self.record),
            "score": self.score,
            "is_single_token": self.record.is_single_token
        }



class Scorer(ABC):
    @abstractmethod
    def __call__(self, record: LatentRecord) -> ScorerResult:
        pass
