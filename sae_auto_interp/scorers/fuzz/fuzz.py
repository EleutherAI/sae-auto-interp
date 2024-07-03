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

from .fuzz_prompt import prompt as fuzz_prompt
from .clean_prompt import prompt as clean_prompt
from ... import det_config as CONFIG
from ..scorer import Scorer, ScorerInput
from pydantic import BaseModel

@dataclass
class Sample:
    text: str
    quantile: int
    clean: bool
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

    def create_example(
        self, 
        tokens: List[str], 
        activations,
        n_incorrect=0,
        threshold=0.0,
        highlight=False
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
        
        check = lambda i: activations[i] > threshold \
            or i in random_indices
    
        return self.prepare(
            tokens, 
            check, 
            highlight=highlight
        )
    
    def prepare(self, tokens, check, highlight=False):

        if not highlight:
            return "".join(tokens)

        result = []

        i = 0
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
        quantile: int,
        highlight: bool = False
    ) -> List[Sample]:
        
        samples = []

        for example in batch:

            n_incorrect = 2 if quantile == -1 else 0
            correct = self.create_example(
                example.str_toks, 
                example.activations,
                n_incorrect=n_incorrect,
                threshold=CONFIG.threshold,
                highlight=highlight
            )

            samples.append(
                Sample(
                    text=correct, 
                    quantile=quantile, 
                    clean=highlight,
                    is_correct=True
                )
            )

        return samples
    
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

        return fuzz_prompt(
            examples, 
            explanation
        )


    async def query(
        self, batch: List[Sample], 
        explanation: str, 
        max_retries: int = 3
    ) -> List[Sample]:
        prompt_type = batch[0].clean

        if prompt_type == True:
            prompt = clean_prompt(batch, explanation)
        else:
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

        clean = []
        fuzzed = []

        for highlight in [False, True]:
            
            for quantile, batch in enumerate(scorer_in.test_examples):
                samples = self.create_samples(
                    batch, 
                    quantile=quantile, 
                    highlight=highlight
                )

                if highlight:
                    fuzzed.extend(samples)
                else:
                    clean.extend(samples)

            random_samples = self.create_samples(
                scorer_in.record.random, 
                quantile=-1, 
                highlight=highlight
            )
            
            if highlight:
                fuzzed.extend(random_samples)
            else:
                clean.extend(random_samples)

        random.shuffle(clean)
        random.shuffle(fuzzed)

        clean_batches = [clean[i:i + 10] for i in range(0, len(clean), 10)]
        fuzzed_batches = [fuzzed[i:i + 10] for i in range(0, len(fuzzed), 10)]

        results = await self.process_batches(
            clean_batches + fuzzed_batches, 
            scorer_in.explanation
        )

        return results