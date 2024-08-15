import json
import os
import random
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
    def __call__(self, record: FeatureRecord) -> ExplainerResult:
        pass


async def explanation_loader(record: FeatureRecord, explanation_dir: str) -> ExplainerResult:
    async with aiofiles.open(f'{explanation_dir}/{record.feature}.txt', 'r') as f:
        explanation = json.loads(await f.read())
    
    return ExplainerResult(
        record=record,
        explanation=explanation
    )

async def random_explanation_loader(record: FeatureRecord, explanation_dir: str) -> ExplainerResult:
    explanations = [f for f in os.listdir(explanation_dir) if f.endswith(".txt")]
    if str(record.feature) in explanations:
        explanations.remove(str(record.feature))
    random_explanation = random.choice(explanations)
    async with aiofiles.open(f'{explanation_dir}/{random_explanation}', 'r') as f:
        explanation = json.loads(await f.read())
    
    return ExplainerResult(
        record=record,
        explanation=explanation
    )