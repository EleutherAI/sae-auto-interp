from typing import List

from ...features import Example
from ...oai_autointerp import LogprobFreeExplanationTokenSimulator, simulate_and_score
from ...oai_autointerp.activations.activations import ActivationRecord
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
    ):
        self.client = client
        self.tokenizer = tokenizer

    async def __call__(self, record):
        # Simulate and score the explanation.
        simulator = LogprobFreeExplanationTokenSimulator(
            self.client,
            record.explanation,
        )

        valid_activation_records = self.to_activation_records(record.test)

        result = await simulate_and_score(simulator, valid_activation_records)

        return ScorerResult(
            record=record,
            score=result,
        )

    def to_activation_records(self, examples: List[Example]) -> List[ActivationRecord]:
        return [
            ActivationRecord(
                self.tokenizer.batch_decode(example.tokens), example.activations
            )
            for example in examples
        ]
