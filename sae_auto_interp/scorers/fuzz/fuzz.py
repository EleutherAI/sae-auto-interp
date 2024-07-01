import random
import asyncio
from dataclasses import dataclass
import json
import logging
from typing import List, Tuple
import numpy as np
import torch

from .prompts import get_detection_template
from ... import det_config as CONFIG
from ..scorer import Scorer, ScorerResult, ScorerInput
from ...features.features import FeatureRecord
from pydantic import BaseModel

@dataclass
class Sample:
    text: str
    score: float
    quantile: int
    is_correct: bool = False
    marked: bool = False
    probability: float = -1.0

class ResponseModel(BaseModel):
    example_1: int
    prob_1: float
    example_2: int
    prob_2: float
    example_3: int
    prob_3: float
    example_4: int
    prob_4: float
    example_5: int
    prob_5: float


class FuzzingScorer(Scorer):
    def __init__(self, client, decay_factor=0.5):
        super().__init__(validate=True)
        self.name = "fuzzing"
        self.client = client
        self.decay_factor = decay_factor

    def _calculate_distance_scores(self, activations: torch.Tensor, decay_factor: float = 0.5) -> List[float]:
        # Create indices tensor
        indices = torch.arange(len(activations)).float()
        
        # Create mask for active tokens
        mask = activations > 0
        
        # Find indices of active tokens
        active_indices = indices[mask]
        
        # Calculate distances to all active tokens
        distances = torch.abs(indices.unsqueeze(1) - active_indices.unsqueeze(0))
        
        # Find minimum distance for each token
        min_distances, _ = distances.min(dim=1)
        
        # Apply exponential decay
        scores = torch.exp(-decay_factor * min_distances)
        
        # Set scores for active tokens to their activation values
        scores[mask] = activations[mask]
        
        return scores.tolist()

    def prepare_example(self, tokens: List[str], activations: List[float]) -> str:
        result = []
        i = 0
        while i < len(tokens):
            if activations[i] > 0:
                result.append("<<")
                while i < len(tokens) and activations[i] > 0:
                    result.append(tokens[i])
                    i += 1
                result.append(">>")
            else:
                result.append(tokens[i])
                i += 1
        return "".join(result)
    
    def _preprocess_sample(self, example, max_activation: float ):
        normalized_activations = example.activations/max_activation
        distance_scores = self._calculate_distance_scores(normalized_activations)
        return distance_scores

    def create_samples(self, batch, max_activation, quantile) -> List[Sample]:
        samples = []
        for i, example in enumerate(batch):

            if i < 5:
                distance_scores = self._preprocess_sample(example, max_activation)
                correct = self.prepare_example(example.str_toks, example.activations)
                correct_score = sum(score for score, is_active in zip(distance_scores, example.activations) if is_active > 0)
                samples.append(Sample(text=correct, quantile=quantile, score=correct_score, is_correct=True))

            else:
                distance_scores = self._preprocess_sample(example, max_activation)
                incorrect, incorrect_score = self._create_incorrect_sample(example.str_toks, distance_scores)
                samples.append(Sample(text=incorrect, score=incorrect_score, is_correct=False, quantile=quantile))

        return samples

    def _create_incorrect_sample(self, tokens: List[str], distance_scores) -> Tuple[str, float]:
        num_to_highlight = random.randint(1, max(1, len(tokens) // 3))
        incorrect_highlights = set(random.sample(range(len(tokens)), num_to_highlight))
        
        result = []
        score = 0
        i = 0
        while i < len(tokens):
            if i in incorrect_highlights:
                result.append("<<")
                while i < len(tokens) and i in incorrect_highlights:
                    result.append(tokens[i])
                    score += distance_scores[i]
                    i += 1
                result.append(">>")
            else:
                result.append(tokens[i])
                i += 1

        return "".join(result), score

    async def process_batches(self, batches: List[List[Sample]], explanation: str) -> List[Sample]:
        tasks = [self.query(batch, explanation) for batch in batches]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

    async def query(self, batch: List[Sample], explanation: str, max_retries: int = 3) -> List[Sample]:
        examples = "\n".join(f"Example {i}: {sample.text}" for i, sample in enumerate(batch))
        prompt = get_detection_template(examples, explanation)

        for attempt in range(max_retries):
            try:
                response = await self.client.async_generate(
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
            sample.marked = selections[f"example_{i+1}"] == 1
            sample.probability = selections[f"prob_{i+1}"]
        
        print(selections)
        return batch

    async def __call__(self, scorer_in: ScorerInput) -> ScorerResult:
        random.seed(CONFIG.seed)

        samples = []
        for quantile, batch in enumerate(scorer_in.test_examples):
            samples.extend(
                self.create_samples(
                    max_activation=scorer_in.record.max_activation(), 
                    batch=batch, 
                    quantile=quantile
                )
            )

        random.shuffle(samples)
        sample_batches = [samples[i:i + 5] for i in range(0, len(samples), 5)]

        results = await self.process_batches(sample_batches, scorer_in.explanation)

        return results
        # return ScorerResult(input="", response="", score=results)