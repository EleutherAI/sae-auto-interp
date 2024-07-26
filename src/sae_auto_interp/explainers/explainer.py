from abc import ABC, abstractmethod
from typing import NamedTuple

import aiofiles

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
    ) -> ExplainerResult:
        pass


async def explanation_loader(record: FeatureRecord, explanation_dir: str) -> str:
    async with aiofiles.open(f'{explanation_dir}/{record.feature}.txt', 'r') as f:
        explanation = await f.read()
    
    return ExplainerResult(
        record=record,
        explanation=explanation
    )

