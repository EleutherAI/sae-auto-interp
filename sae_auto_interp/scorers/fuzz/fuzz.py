import random
import asyncio
from dataclasses import dataclass
import json
import logging
from typing import List, Tuple
import numpy as np

from .prompts import get_detection_template
from ... import det_config as CONFIG
from ..scorer import Scorer, ScorerResult, ScorerInput
from ...features.features import Feature, FeatureRecord
from pydantic import BaseModel

@dataclass
class Sample:
    text: str
    score: float
    is_correct: bool = False
    marked: bool = False

class ResponseModel(BaseModel):
    example_1: int
    example_2: int
    example_3: int
    example_4: int
    example_5: int

class FuzzingScorer(Scorer):
    def __init__(self, client, decay_factor=0.5):
        super().__init__(validate=True)
        self.name = "fuzzing"
        self.client = client
        self.decay_factor = decay_factor

    def _calculate_distance_scores(self, normalized_activations: List[float]) -> List[float]:
        n = len(normalized_activations)
        scores = np.zeros(n)
        last_active = -1
        
        for i in range(n):
            if normalized_activations[i] > 0:
                scores[i] = normalized_activations[i]
                last_active = i
            elif last_active != -1:
                scores[i] = np.exp(-self.decay_factor * (i - last_active))
        
        # Handle right-to-left pass
        last_active = -1
        for i in range(n - 1, -1, -1):
            if normalized_activations[i] > 0:
                last_active = i
            elif last_active != -1:
                scores[i] = max(scores[i], np.exp(-self.decay_factor * (last_active - i)))
        
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

    def create_samples(self, example: FeatureRecord, distance_scores: List[float]) -> List[Sample]:
        correct = self.prepare_example(example.str_toks, example.activations)
        correct_score = sum(score for score, is_active in zip(distance_scores, example.activations) if is_active > 0)
        samples = [Sample(text=correct, score=correct_score, is_correct=True)]

        for _ in range(4):
            incorrect, incorrect_score = self._create_incorrect_sample(example.str_toks, distance_scores )
            samples.append(Sample(text=incorrect, score=incorrect_score, is_correct=False))

        random.shuffle(samples)
        return samples

    def _create_incorrect_sample(self, tokens: List[str], distance_scores) -> str:
        num_to_highlight = random.randint(1, max(1, len(tokens) // 3))
        incorrect_highlights = set(random.sample(range(len(tokens)), num_to_highlight))
        
        result = []
        score = 0
        i  = 0
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

        for sample, (_, mark) in zip(batch, selections.items()):
            sample.marked = (mark == 1)

        return batch

    def __call__(self, scorer_in: ScorerInput) -> ScorerResult:
        random.seed(CONFIG.seed)
        
        sample_batches = []
        for example in scorer_in.test_examples:
            normalized_activations = [act / example.max_activation for act in example.activations]
            distance_scores = self._calculate_distance_scores(normalized_activations)
            sample_batches.append(self.create_samples(example, distance_scores))

        results = asyncio.run(self.process_batches(sample_batches, scorer_in.explanation))

        return ScorerResult(input="", response="", score=results)