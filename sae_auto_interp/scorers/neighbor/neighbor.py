import random
import asyncio
from typing import List, Tuple
import torch
from math import ceil

from .clean_prompt import prompt as clean_prompt
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
    distance: float
    neighbor_index: int
    predicted: bool = None

    @staticmethod
    def _prepare_samples(
        examples: List, 
        distance: float,
        neighbor_index: int,
        ground_truth: bool, 
        tokenizer,
    ):
        samples = []

        for example in examples:
            example.decode(tokenizer)

            samples.append(
                Sample(
                    text=example.text,
                    ground_truth = ground_truth,
                    distance = distance,
                    neighbor_index = neighbor_index
                )
            )

        return samples
    
    def default(self):
        return {
            "ground_truth": self.ground_truth,
            "distance": self.distance,
            "neighbor_index": self.neighbor_index,
            "predicted": self.predicted
        }

class NeighborScorer(Scorer):
    name = "neighbor"

    def __init__(
        self, 
        client: Client, 
        tokenizer,
        echo: bool = False, 
        temperature: float = 0.0,
        max_tokens: int = 2,
        batch_size: int = 1,
        n_test: int = 5,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.echo = echo

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        
        self.n_test = n_test

    async def __call__(
        self, 
        scorer_in: ScorerInput,
    ) -> List[Sample]:

        samples = self._prepare(
            scorer_in.test_examples,
            scorer_in.record.neighbors
        )

        # Generate responses
        results = await self.process_batches(
            samples,
            scorer_in.explanation
        )

        return results
    
    def _prepare(self, test_examples, neighbors):

        samples = Sample._prepare_samples(
            test_examples,
            0.0,
            0,
            True,
            self.tokenizer
        )

        for i, (distance, neighbor)\
            in enumerate(neighbors.items()):

            # Neighbor was probably too sparse to cache
            if neighbor is None: 
                continue

            examples = random.sample(
                neighbor.examples,
                self.n_test
            )

            samples.extend(
                Sample._prepare_samples(
                    examples,
                    distance,
                    i + 1,
                    False,
                    self.tokenizer
                )
            )
        
        return [
            samples[i:i + self.batch_size] 
            for i in range(0, len(samples), self.batch_size)
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
            item.default()
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

        return clean_prompt(
            explanation=explanation,
            examples=examples,
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

        if self.batch_size > 1:
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