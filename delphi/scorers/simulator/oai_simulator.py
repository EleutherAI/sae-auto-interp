from typing import List

from ...features import Example
from .oai_autointerp import (
    ActivationRecord,
    ExplanationNeuronSimulator,
    LogprobFreeExplanationTokenSimulator,
    simulate_and_score,
)
from ..scorer import Scorer, ScorerResult


class OpenAISimulator(Scorer):
    """
    Simple wrapper for the LogProbFreeExplanationTokenSimulator.
    """

    name = "simulator"

    def __init__(
        self,
        client,
        tokenizer,
        all_at_once=True,
    ):
        self.client = client
        self.tokenizer = tokenizer  
        self.all_at_once = all_at_once

    async def __call__(self, record):
        # Simulate and score the explanation.
        cls = (
            ExplanationNeuronSimulator
            if self.all_at_once
            else LogprobFreeExplanationTokenSimulator
        )
        simulator = cls(
            self.client,
            record.explanation,
        )

        valid_activation_records = self.to_activation_records(record.test)
        if len(record.random_examples) > 0:
            non_activation_records = self.to_activation_records([record.random_examples])
        else:
            non_activation_records = []

        result = await simulate_and_score(simulator, valid_activation_records, non_activation_records)

        return ScorerResult(
            record=record,
            score=result,
        )

    def to_activation_records(self, examples: list[Example]) -> list[ActivationRecord]:
        return [
            [
                ActivationRecord(
                    self.tokenizer.batch_decode(example.tokens), example.normalized_activations.half()
                )
                for example in quantiles
            ]
            for quantiles in examples
        ]
        
