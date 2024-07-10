from typing import List

from ...features import Example
from ...oai_autointerp.activations.activations import ActivationRecord
from ...oai_autointerp import LogprobFreeExplanationTokenSimulator, simulate_and_score
from ..scorer import Scorer, ScorerInput

class OpenAISimulator(Scorer):
    """
    Simple wrapper for the LogProbFreeExplanationTokenSimulator.
    """
    def __init__(
        self,
        client,
    ):
        self.client = client
        self.name = "simulator"

    async def __call__(
        self,
        scorer_in: ScorerInput,
        echo=False
    ):
        # Simulate and score the explanation.
        simulator = LogprobFreeExplanationTokenSimulator(
            self.client,
            scorer_in.explanation,
        )

        valid_activation_records = self.to_activation_records(
            scorer_in.test_examples
        )

        result = await simulate_and_score(
            simulator, 
            valid_activation_records
        )

        return result

    def to_activation_records(
        self,
        examples: List[Example]
    ) -> List[ActivationRecord]:
        return [ 
            ActivationRecord(
                example.str_toks, 
                example.activations
            ) 
            for example in examples
        ]
    