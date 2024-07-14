import random
import asyncio
from typing import List, Tuple
import torch
from math import ceil

from .prompts.fuzz_prompt import prompt as fuzz_prompt
from .prompts.clean_prompt import prompt as clean_prompt
from .schema import create_response_model
from ..scorer import Scorer, ScorerInput
from ...features import Example
from ...clients.client import Client
from dataclasses import dataclass

import json


@dataclass
class Sample:
    text: str
    ground_truth: bool
    predicted: bool = None
    distance: float = None
    neighbor_index: int = None

class NeighborScorer(Scorer):
    name = "neighbor"

    def __init__(
        self, 
        client: Client, 
        echo: bool = False, 
        temperature: float = 0.0,
        max_tokens: int = 200,
        batch_size: int = 10,
    ):
        self.client = client
        self.echo = echo

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size

    async def __call__(
        self, 
        scorer_in: ScorerInput
    ) -> List[Sample]:
        
        # Build clean and fuzzed batches
        clean_batches, fuzzed_batches = self._prepare(
            test_batches=scorer_in.test_examples, 
            random_examples=scorer_in.random_examples,
            incorrect_examples=scorer_in.extra_examples,
            avg_acts=scorer_in.record.average_n_activations
        )

        # Generate responses
        results = await self.process_batches(
            clean_batches + fuzzed_batches, 
            scorer_in.explanation
        )

        return results
        
    def _prepare(self, test_batches, random_examples, incorrect_examples, avg_acts):
        def create_samples(examples, quantile, highlight, ground_truth=True):

            n_incorrect = ceil(avg_acts) \
                if highlight and not ground_truth else 0
            
            return [
                Sample(
                    example=example,
                    quantile=quantile,
                    highlighted=highlight,
                    ground_truth=ground_truth,
                    n_incorrect=n_incorrect,
                    id=hash(example)
                )
                for example in examples
            ]

        clean = []
        fuzzed = []

        for quantile, batch in enumerate(test_batches):
            clean.extend(create_samples(batch, quantile, False))
            fuzzed.extend(create_samples(batch, quantile, True))

        clean.extend(create_samples(random_examples, -1, False, False))
        fuzzed.extend(create_samples(incorrect_examples, -1, True, False))

        random.shuffle(clean)
        random.shuffle(fuzzed)

        return self._batch(clean), self._batch(fuzzed)
    
    def _batch(self, arr):
        return [
            arr[i:i + self.batch_size] 
            for i in range(0, len(arr), self.batch_size)
        ]
    
    async def process_batches(
        self, 
        batches: List[List[Sample]], 
        explanation: str
    ) -> List[Sample]:
        # Create a list of tasks to be executed concurrently
        tasks = [
            self.query(batch, explanation) 
            for batch in batches
        ]

        # Execute the tasks concurrently
        results = await asyncio.gather(*tasks)

        # Return a flattened list of samples
        return [
            item.default(echo=self.echo)
            for sublist in results 
            for item in sublist
        ]


    def build_prompt(
        self, 
        batch: List[Sample], 
        explanation: str
    ) -> str:
        examples = "\n".join(
            f"Example {i}: {sample.text}" 
            for i, sample in enumerate(batch)
        )

        prompt = fuzz_prompt if batch[0].highlighted else clean_prompt

        return prompt(
            explanation=explanation,
            examples=examples,
            n_test=self.n_few_shots
        )

    async def query(
        self, 
        batch: List[Sample], 
        explanation: str
    ) -> List[Sample]:
        prompt = self.build_prompt(batch, explanation)

        generation_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if len(batch) > 1:
            selections = await self.client.generate(
                prompt,
                schema=create_response_model(len(batch)),
                **generation_kwargs
            )

            for i, sample in enumerate(batch):
                sample.predicted = selections[f"example_{i}"] == 1

        else:
            selections = await self.client.generate(
                prompt,
                **generation_kwargs
            )

            batch[0].predicted = int(selections[-1]) == 1

        return batch