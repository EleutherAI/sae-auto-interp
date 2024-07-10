import random
import asyncio
from typing import List, Tuple
import torch
from math import ceil

from .sample import Sample
from .fuzz_prompt import prompt as fuzz_prompt
from .clean_prompt import prompt as clean_prompt
from .schema import create_response_model
from ... import det_config as CONFIG
from ..scorer import Scorer, ScorerInput

import json


class FuzzingScorer(Scorer):
    def __init__(self, client, echo=False, get_prompts=False, n_few_shots=-1):
        self.name = "fuzz"
        self.client = client
        self.echo = echo
        self.n_few_shots = n_few_shots
        self.get_prompts = get_prompts
    
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

        if self.get_prompts:
            return clean_batches, fuzzed_batches
        
        # Generate responses
        results = await self.process_batches(
            clean_batches + fuzzed_batches, 
            scorer_in.explanation
        )

        return results
    
    def _prepare(
        self, 
        test_batches,
        random_examples,
        incorrect_examples,
        avg_acts
    ):  
        def _prepare_batch(arr, batch, quantile, highlight, ground_truth=True):
            
            n_incorrect = ceil(avg_acts) \
                if (
                    quantile == -1 
                    and highlight
                ) else 0

            # Append to a respective list
            for example in batch:
                arr.append(
                    Sample(
                        example=example,
                        quantile=quantile,
                        highlighted=highlight,
                        ground_truth=ground_truth,
                        n_incorrect=n_incorrect,
                        id=hash(example.text),
                        echo=self.echo
                    )
                )

        clean = []
        fuzzed = []

        for quantile, batch in enumerate(test_batches):
            _prepare_batch(clean, batch, quantile, highlight=False)
            _prepare_batch(fuzzed, batch, quantile, highlight=True)

        # Add random examples to the clean batch
        _prepare_batch(clean, random_examples, -1, highlight=False, ground_truth=False)
        # Add incorrectly predicted examples to the fuzzed batch
        _prepare_batch(fuzzed, incorrect_examples, -1, highlight=True, ground_truth=False)

        random.shuffle(clean)
        random.shuffle(fuzzed)

        return self._batch(clean), self._batch(fuzzed)
    
        
    def _batch(self, arr):
        return [
            arr[i:i + CONFIG.batch_size] 
            for i in range(0, len(arr), CONFIG.batch_size)
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
        response_model = create_response_model(len(batch))

        print(prompt)
        selections = await self.client.generate(
            prompt,
            max_tokens=CONFIG.max_tokens,
            temperature=CONFIG.temperature,
        )

        # for i, sample in enumerate(batch):
        #     sample.predicted = selections[f"example_{i}"] == 1

        batch[0].predicted = int(selections[-1]) == 1
        
        return batch