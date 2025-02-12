import json
import os
import random
from abc import ABC, abstractmethod
from typing import NamedTuple

import aiofiles

from ..latents.latents import LatentRecord


class ExplainerResult(NamedTuple):
    record: LatentRecord
    """Latent record passed through to scorer."""

    explanation: str
    """Generated explanation for latent."""


class Explainer(ABC):
    @abstractmethod
    def __call__(self, record: LatentRecord) -> ExplainerResult:
        pass


async def explanation_loader(
    record: LatentRecord, explanation_dir: str
) -> ExplainerResult:
    async with aiofiles.open(f"{explanation_dir}/{record.latent}.txt", "r") as f:
        explanation = json.loads(await f.read())

    return ExplainerResult(record=record, explanation=explanation)


async def random_explanation_loader(
    record: LatentRecord, explanation_dir: str
) -> ExplainerResult:
    explanations = [f for f in os.listdir(explanation_dir) if f.endswith(".txt")]
    if str(record.latent) in explanations:
        explanations.remove(str(record.latent))
    random_explanation = random.choice(explanations)
    async with aiofiles.open(f"{explanation_dir}/{random_explanation}", "r") as f:
        explanation = json.loads(await f.read())

    return ExplainerResult(record=record, explanation=explanation)
