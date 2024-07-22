import random
import asyncio
from typing import List
import torch
from math import ceil

from transformers import PreTrainedTokenizer

from .classifier import Classifier
from .sample import examples_to_samples, Sample, ClassifierOutput
from .prompts.fuzz_prompt import prompt
from ..scorer import Scorer, ScorerResult
from ...clients.client import Client
from ...features import FeatureRecord


class FuzzingScorer(Classifier, Scorer):
    name = "fuzz"

    def __init__(
        self, 
        client: Client, 
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 1,
        threshold: float = 0.3,
        **generation_kwargs
    ):
        self.client = client
        self.tokenizer = tokenizer  

        self.batch_size = batch_size
        self.threshold = threshold

        self.generation_kwargs = generation_kwargs
    
    def average_n_activations(self, examples) -> float:
        return sum(
            len(torch.nonzero(example.activations)) 
            for example in examples
        ) / len(examples)
        
    def _prepare(
        self, 
        record: FeatureRecord
    ) -> List[List[Sample]]:
        """
        Prepare and shuffle a list of samples for classification.
        """

        samples = examples_to_samples(
            record.random_examples,
            distance = -1,
            ground_truth = False,
            tokenizer = self.tokenizer
        )

        for i, examples in enumerate(record.test):

            samples.extend(
                examples_to_samples(
                    examples,
                    distance = i + 1,
                    ground_truth = True,
                    tokenizer = self.tokenizer
                )
            )

        random.shuffle(samples)
        
        return [
            samples[i : i + self.batch_size] 
            for i in range(0, len(samples), self.batch_size)
        ]
    
    def _batch(self, arr):
        return [
            arr[i:i + self.batch_size] 
            for i in range(0, len(arr), self.batch_size)
        ]
    
    async def _query(
        self, 
        batches: List[List[Sample]], 
        explanation: str
    ) -> List[Sample]:
        # Create a list of tasks to be executed concurrently
        tasks = [
            self._generate(explanation, batch) 
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

    async def _generate(
        self, 
        explanation: str,
        batch: List[Sample]
    ) -> List[Sample]:
        prompt = self.build_prompt(explanation, batch)
        
        selections = await self.client.generate(
            prompt,
            **self.generation_kwargs
        )

        for i, sample in enumerate(batch):
            sample.predicted = selections[i] == 1
        
        return batch

    def build_prompt(
        self, 
        explanation: str,
        batch: List[Sample]
    ) -> str:
        
        examples = "\n".join(
            f"Example {i}: {sample.text}" 
            for i, sample in enumerate(batch)
        )

        return prompt(
            explanation=explanation,
            examples=examples,
        )