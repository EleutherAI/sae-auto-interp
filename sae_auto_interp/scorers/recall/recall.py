import asyncio
from dataclasses import dataclass
from typing import List, NamedTuple

import torch

from .prompt import prompt as clean_prompt
from ..scorer import Scorer, ScorerResult
from ...clients.client import Client, create_response_model


@dataclass
class Sample:
    text: str
    quantile: int
    ground_truth: bool
    predicted: bool = None

    @staticmethod
    def _prepare_samples(
        examples: List, 
        quantile: int,
        ground_truth: bool, 
        tokenizer,
    ):
        samples = []

        for example in examples:

            samples.append(
                Sample(
                    text=tokenizer.decode(example.tokens),
                    quantile=quantile,
                    ground_truth = ground_truth
                )
            )

        return samples
    
    def default(self):
        return {
            "text": self.text,
            "quantile": self.quantile,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
        }

class RecallScorer(Scorer):
    name = "recall"

    def __init__(
        self, 
        client: Client, 
        tokenizer,
        echo: bool = False, 
        temperature: float = 0.0,
        max_tokens: int = 300,
        batch_size: int = 10,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.echo = echo

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size

    async def __call__(
        self, 
        record
    ) -> List[Sample]:

        samples = self._prepare(
            record.test,
            record.random_examples
        )

        # Generate responses
        results = await self.process_batches(
            samples,
            record.explanation
        )

        return ScorerResult(
            record=record,
            score=results
        )
    
    def _prepare(self, activating_examples, incorrect_examples):

        samples = Sample._prepare_samples(
            incorrect_examples,
            -1,
            False,
            self.tokenizer
        )

        for i, examples in enumerate(activating_examples):

            samples.extend(
                Sample._prepare_samples(
                    examples,
                    i + 1,
                    True,
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
            examples=examples
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

        selections = await self.client.generate(
            prompt,
            **generation_kwargs
        )

        for i, sample in enumerate(batch):
            sample.predicted = selections[i] == 1

        return batch