from dataclasses import dataclass
from typing import List

from ...autoencoders.features import FeatureRecord, Example

from ...oai_autointerp.activations.activations import ActivationRecord
from ...oai_autointerp import TokenActivationPairExplainer

# from ... import explainer_cfg
import asyncio

from ..explainer import Explainer, ExplainerInput, ExplainerResult
import time

class OpenAIExplainer(Explainer):
    def __init__(self, model):
        self.name = "oai"
        self.model = model

    def __call__(
        self,
        explainer_in: ExplainerInput,
        echo=False
    ) :
        records = self.to_activation_records(explainer_in.train_examples)

        explainer = TokenActivationPairExplainer(
            model_name=self.model,
            max_concurrent=1,
        )

        prompt, response, explanations = asyncio.run(
            explainer.generate_explanations(
                all_activation_records=records,
                max_activation=explainer_in.record.max_activation,
                num_samples=1,
                echo=echo
            )
        )

        # Input and response here don't have much info/variance.
        return ExplainerResult(
            explainer_type=self.name,
            input="",
            response="",
            explanation=explanations[0]
        )

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