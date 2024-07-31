import asyncio
import json
import random
import re
from abc import abstractmethod
from typing import List

from transformers import PreTrainedTokenizer

from ...clients.client import Client
from ...features import FeatureRecord
from ..scorer import Scorer, ScorerResult
from .sample import ClassifierOutput, Sample


class Classifier(Scorer):
    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer,
        verbose: bool,
        batch_size: int,
        **generation_kwargs,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.verbose = verbose

        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs

    async def __call__(
        self,
        record: FeatureRecord,
    ) -> List[ClassifierOutput]:
        samples = self._prepare(record)

        random.shuffle(samples)

        results = await self._query(
            record.explanation,
            self._batch(samples),
        )

        return ScorerResult(record=record, score=results)

    @abstractmethod
    def _prepare(self, record: FeatureRecord) -> List[List[Sample]]:
        pass

    async def _query(
        self,
        explanation: str,
        batches: List[List[Sample]],
    ) -> List[Sample]:
        """
        Send and gather batches of samples to the model.
        """

        tasks = [self._generate(explanation, batch) for batch in batches]

        results = await asyncio.gather(*tasks)

        return sum(results, [])

    async def _generate(
        self, explanation: str, batch: List[Sample]
    ) -> List[ClassifierOutput]:
        """
        Generate predictions for a batch of samples.
        """

        prompt = self._build_prompt(explanation, batch)

        selections = await self.client.generate(prompt, **self.generation_kwargs)
        array = self._parse(selections)

        # print(selections, array)

        results = []

        for i, sample in enumerate(batch):
            result = sample.data
            prediction = array[i] == 1

            result.prediction = prediction == result.ground_truth

            results.append(result)

            if self.verbose:
                result.text = sample.text

        return results

    def _parse(self, string):
        pattern = r"\[.*?\]"
        match = re.search(pattern, string)

        try:
            array = json.loads(match.group(0))
            assert len(array) == self.batch_size
            return array
        except (json.JSONDecodeError, AssertionError):
            return [-1] * self.batch_size

    def _build_prompt(
        self,
        explanation: str,
        batch: List[Sample],
    ) -> str:
        """
        Prepare prompt for generation.
        """

        examples = "\n".join(
            f"Example {i}: {sample.text}" for i, sample in enumerate(batch)
        )

        return self.prompt(explanation=explanation, examples=examples)

    def _batch(self, samples):
        return [
            samples[i : i + self.batch_size]
            for i in range(0, len(samples), self.batch_size)
        ]
