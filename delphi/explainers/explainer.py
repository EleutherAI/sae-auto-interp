import json
import os
import random
from abc import ABC, abstractmethod
from typing import NamedTuple
import re

import aiofiles

from ..latents.latents import LatentRecord
from ..logger import logger

class ExplainerResult(NamedTuple):
    record: LatentRecord
    """Latent record passed through to scorer."""

    explanation: str
    """Generated explanation for latent."""


class Explainer(ABC):
    async def __call__(self, record):

        messages = self._build_prompt(record.train)

        response = await self.client.generate(
            messages, temperature=self.temperature, **self.generation_kwargs
        )

        try:
            explanation = self.parse_explanation(response.text)
            if self.verbose:
                return (
                    messages[-1]["content"],
                    response,
                    ExplainerResult(record=record, explanation=explanation),
                )

            return ExplainerResult(record=record, explanation=explanation)
        except Exception as e:
            logger.error(f"Explanation parsing failed: {e}")
            return ExplainerResult(record=record, explanation="Explanation could not be parsed.")

    def parse_explanation(self, text: str) -> str:
        try:
            match = re.search(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL)
            return match.group(1).strip() if match else "Explanation could not be parsed."
        except Exception as e:
            logger.error(f"Explanation parsing regex failed: {e}")
            raise

    def _highlight(self, index, example):
        # result = f"Example {index}: "
        result = ""
        threshold = example.max_activation * self.threshold
        if self.tokenizer is not None:
            str_toks = self.tokenizer.batch_decode(example.tokens)
            example.str_toks = str_toks
        else:
            str_toks = example.tokens
            example.str_toks = str_toks
        activations = example.activations

        def check(i):
            return activations[i] > threshold

        i = 0
        while i < len(str_toks):
            if check(i):
                # result += "<<"

                while i < len(str_toks) and check(i):
                    result += str_toks[i]
                    i += 1
                # result += ">>"
            else:
                # result += str_toks[i]
                i += 1

        return "".join(result)

    def _join_activations(self, example):
        activations = []

        for i, activation in enumerate(example.activations):
            if activation > example.max_activation * self.threshold:
                activations.append((example.str_toks[i], int(example.normalized_activations[i])))

        acts = ", ".join(f'("{item[0]}" : {item[1]})' for item in activations)

        return "Activations: " + acts


async def explanation_loader(record: LatentRecord, explanation_dir: str) -> ExplainerResult:
    async with aiofiles.open(f'{explanation_dir}/{record.latent}.txt', 'r') as f:
        explanation = json.loads(await f.read())
    
    return ExplainerResult(
        record=record,
        explanation=explanation
    )

async def random_explanation_loader(record: LatentRecord, explanation_dir: str) -> ExplainerResult:
    explanations = [f for f in os.listdir(explanation_dir) if f.endswith(".txt")]
    if str(record.latent) in explanations:
        explanations.remove(str(record.latent))
    random_explanation = random.choice(explanations)
    async with aiofiles.open(f'{explanation_dir}/{random_explanation}', 'r') as f:
        explanation = json.loads(await f.read())
    
    return ExplainerResult(
        record=record,
        explanation=explanation
    )