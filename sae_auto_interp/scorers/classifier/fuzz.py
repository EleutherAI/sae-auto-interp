from math import ceil
from typing import List

import torch
from transformers import PreTrainedTokenizer

from ...clients.client import Client
from ...features import FeatureRecord
from ..scorer import Scorer
from .classifier import Classifier
from .prompts.fuzz_prompt import prompt
from .sample import Sample, examples_to_samples


class FuzzingScorer(Classifier, Scorer):
    name = "fuzz"

    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer,
        verbose: bool = False,
        batch_size: int = 1,
        threshold: float = 0.3,
        **generation_kwargs,
    ):
        super().__init__(
            client=client,
            tokenizer=tokenizer,
            verbose=verbose,
            batch_size=batch_size,
            **generation_kwargs,
        )

        self.threshold = threshold
        self.prompt = prompt

    def average_n_activations(self, examples) -> float:
        avg = sum(
            len(torch.nonzero(example.activations)) for example in examples
        ) / len(examples)

        return ceil(avg)

    def _prepare(self, record: FeatureRecord) -> List[List[Sample]]:
        """
        Prepare and shuffle a list of samples for classification.
        """

        defaults = {
            "highlighted": True,
            "tokenizer": self.tokenizer,
        }

        n_incorrect = self.average_n_activations(record.extra_examples)

        samples = examples_to_samples(
            record.extra_examples,
            distance=-1,
            ground_truth=False,
            n_incorrect=n_incorrect,
            **defaults,
        )

        for i, examples in enumerate(record.test):
            samples.extend(
                examples_to_samples(
                    examples,
                    distance=i + 1,
                    ground_truth=True,
                    n_incorrect=0,
                    **defaults,
                )
            )

        return samples
