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


class FuzzingScorer(Scorer):
    def __init__(self, client):
        self.name = "fuzz"
        self.client = client
    
    async def __call__(
        self, 
        scorer_in: ScorerInput
    ) -> List[Sample]:
        
        self.avg_acts = self.average_n_acts(scorer_in.record.examples[:150])
        
        # Build clean and fuzzed batches
        clean_batches, fuzzed_batches = self._prepare(
            test_batches=scorer_in.test_examples, 
            random_examples=scorer_in.record.random,
            incorrect_examples=scorer_in.record.extra,
        )

        # Generate responses
        results = await self.process_batches(
            clean_batches + fuzzed_batches, 
            scorer_in.explanation
        )

        return results
    
    def average_n_acts(self, examples):
        return sum(
            torch.count_nonzero(example.activations)
            for example in examples
        ) / len(examples)

    def _prepare(
        self, 
        test_batches,
        random_examples,
        incorrect_examples
    ):  
        def _prepare_batch(arr, batch, quantile, highlight, activates=True):
            # If the example doesn't activate, its a random excerpt
            if batch[0].max_activation == 0.0:
                quantile = -1

            n_incorrect = 0
            if quantile == -1 and highlight:
                n_incorrect = ceil(self.avg_acts.item())
            
            # Append to a respective list
            for example in batch:
                arr.append(
                    Sample(
                        example=example,
                        quantile=quantile,
                        highlighted=highlight,
                        activates=activates,
                        n_incorrect=n_incorrect
                    )
                )

        clean = []
        fuzzed = []

        for quantile, batch in enumerate(test_batches):
            _prepare_batch(clean, batch, quantile, highlight=False)
            _prepare_batch(fuzzed, batch, quantile, highlight=True)

        _prepare_batch(clean, random_examples, -1, highlight=False, activates=False)
        _prepare_batch(fuzzed, incorrect_examples, -1, highlight=True, activates=False)

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

        # for t in batches:
        #     print(len(t))
        
        # return

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
            examples=examples
        )

    async def query(
        self, 
        batch: List[Sample], 
        explanation: str
    ) -> List[Sample]:
        prompt = self.build_prompt(batch, explanation)
        response_model = create_response_model(len(batch))

        selections = await self.client.generate(
            prompt,
            max_tokens=100,
            temperature=0.0,
            schema=response_model.model_json_schema()
        )

        for i, sample in enumerate(batch):
            sample.marked = selections[f"example_{i}"] == 1
        
        return batch