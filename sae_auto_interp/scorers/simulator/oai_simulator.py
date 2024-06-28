from dataclasses import dataclass
from typing import List

from ...autoencoders.features import FeatureRecord, Example
from ...oai_autointerp.activations.activations import ActivationRecord
from ...oai_autointerp import LogprobFreeExplanationTokenSimulator, simulate_and_score
from ..scorer import Scorer, ScorerInput, ScorerResult
import asyncio

class OpenAISimulator(Scorer):
    def __init__(
        self,
        model,
    ):
        super().__init__(validate=True)
        self.name = "simulator"
        self.model = model

    def __call__(
        self,
        scorer_in: ScorerInput,
        echo=False
    ):
        # Simulate and score the explanation.
        simulator = LogprobFreeExplanationTokenSimulator(
            self.model,
            scorer_in.explanation,
        )

        valid_activation_records = self.to_activation_records(
            scorer_in.test_examples
        )

        result = asyncio.run(
            simulate_and_score(
                simulator, 
                valid_activation_records, 
                echo=echo
            )
        )

        return ScorerResult(*result)

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
    