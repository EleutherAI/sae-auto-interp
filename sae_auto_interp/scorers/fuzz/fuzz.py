import random
import asyncio
from dataclasses import dataclass
import json
import logging
from typing import List, Tuple
from ...features.features import Example
import numpy as np
import torch
from torch import Tensor

from .prompts import get_detection_template
from ... import det_config as CONFIG
from ..scorer import Scorer, ScorerInput
from pydantic import BaseModel

@dataclass
class Sample:
    text: str
    quantile: int
    n_incorrect: int
    is_correct: bool = False
    marked: bool = False

class ResponseModel(BaseModel):
    example_1: int
    example_2: int
    example_3: int
    example_4: int
    example_5: int
    example_6: int
    example_7: int
    example_8: int
    example_9: int
    example_10: int


class FuzzingScorer(Scorer):
    def __init__(self, client):
        self.name = "fuzzing"
        self.client = client

    def highlight_example(
        self, 
        tokens: List[str], 
        activations: Tensor[float],
        n_incorrect=0,
        threshold=0.0
    ) -> str:
        threshold = threshold * activations.max()
        below_threshold = torch.nonzero(
            activations <= threshold
        ).squeeze()

        # Random sampling in a loop is relatively slow
        # there might be a better way to do this
        random.seed(CONFIG.seed)
        random_indices = set(
            random.sample(
                below_threshold.tolist(),
                n_incorrect
            )
        )
        
        result = []
        i = 0
        
        check = lambda i: activations[i] > threshold \
            or i in random_indices

        while i < len(tokens):
            if check(i):
                result.append("<<")

                while (
                    i < len(tokens) 
                    and check(i)
                ):
                    result.append(tokens[i])
                    i += 1

                result.append(">>")
            else:
                result.append(tokens[i])
                i += 1

        return "".join(result)
    
    def create_samples(
        self, 
        batch: List[Example], 
        max_activation: float, 
        quantile: int
    ) -> List[Sample]:
        
        samples = []

        for n_incorrect in [0,3]:
            mini_batch = []
            for example in batch:
                normalized_activations = example.activations / max_activation

                correct = self.highlight_example(
                    example.str_toks, 
                    normalized_activations,
                    n_incorrect=n_incorrect,
                    threshold=CONFIG.threshold
                )

                is_correct = n_incorrect == 0
                
                mini_batch.append(
                    Sample(
                        text=correct, 
                        quantile=quantile, 
                        n_incorrect=n_incorrect,
                        is_correct=is_correct
                    )
                )

            samples.append(mini_batch)


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
            item 
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

        return get_detection_template(
            examples, 
            explanation
        )


    async def query(
        self, batch: List[Sample], 
        explanation: str, 
        max_retries: int = 3
    ) -> List[Sample]:
        prompt = self.build_prompt(batch, explanation)

        for attempt in range(max_retries):
            try:
                response = await self.client.generate(
                    prompt,
                    max_tokens=100,
                    temperature=0.0,
                    schema=ResponseModel.model_json_schema()
                )
                selections = json.loads(response)
                break
            except json.JSONDecodeError:
                logging.warning(f"Attempt {attempt + 1}: Invalid JSON response, retrying...")
                if attempt == max_retries - 1:
                    logging.error(f"Max retries reached. Last response: {response}")
                    raise
                await asyncio.sleep(1)

        for i, sample in enumerate(batch):
            sample.marked = selections[f"example_{i + 1}"] == 1
        
        return batch
    

    async def __call__(
        self, 
        scorer_in: ScorerInput
    ) -> List[Sample]:

        random.seed(CONFIG.seed)

        samples = []
        for quantile, batch in enumerate(scorer_in.test_examples):
            samples.extend(
                self.create_samples(
                    max_activation=scorer_in.record.max_activation, 
                    batch=batch, 
                    quantile=quantile
                )
            )

        random.shuffle(samples)
        sample_batches = [samples[i:i + 5] for i in range(0, len(samples), 5)]

        results = await self.process_batches(sample_batches, scorer_in.explanation)

        return results