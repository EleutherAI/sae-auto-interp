from abc import ABC, abstractmethod
from typing import NamedTuple
import json

import aiofiles

from ..features.features import FeatureRecord


class ExplainerResult(NamedTuple):
    record: FeatureRecord
    """Feature record passed through to scorer."""

    explanation: str
    """Generated explanation for feature."""


class Explainer(ABC):
    @abstractmethod
    def __call__(self, record: FeatureRecord) -> ExplainerResult:
        pass


<<<<<<< HEAD:src/sae_auto_interp/explainers/explainer.py
async def explanation_loader(record: FeatureRecord, explanation_dir: str) -> ExplainerResult:
    async with aiofiles.open(f'{explanation_dir}/{record.feature}.txt', 'r') as f:
        explanation = json.loads(await f.read())["explanation"]
    
    return ExplainerResult(
        record=record,
        explanation=explanation
    )
=======
async def explanation_loader(record: FeatureRecord, explanation_dir: str) -> str:
    async with aiofiles.open(f"{explanation_dir}/{record.feature}.txt", "r") as f:
        explanation = await f.read()
>>>>>>> ca973f5a5f4c1feaafaf0dae94c9a3f068104774:sae_auto_interp/explainers/explainer.py

    return ExplainerResult(record=record, explanation=explanation)
