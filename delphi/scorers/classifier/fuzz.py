from math import ceil

import torch

from ...clients.client import Client
from ...latents import LatentRecord
from ...latents.latents import ActivatingExample
from ..scorer import Scorer
from .classifier import Classifier
from .prompts.fuzz_prompt import prompt
from .sample import Sample, examples_to_samples


class FuzzingScorer(Classifier, Scorer):
    name = "fuzz"

    def __init__(
        self,
        client: Client,
        verbose: bool = False,
        n_examples_shown: int = 1,
        threshold: float = 0.3,
        log_prob: bool = False,
        temperature: float = 0.0,
        **generation_kwargs,
    ):
        """
        Initialize a FuzzingScorer.

        Args:
            client: The client to use for generation.
            tokenizer: The tokenizer used to cache the tokens
            verbose: Whether to print verbose output.
            n_examples_shown: The number of examples to show in the prompt,
                        a larger number can both leak information and make
                        it harder for models to generate anwers in the correct format.
            log_prob: Whether to use log probabilities to allow for AUC calculation.
            generation_kwargs: Additional generation kwargs.
        """
        super().__init__(
            client=client,
            verbose=verbose,
            n_examples_shown=n_examples_shown,
            log_prob=log_prob,
            temperature=temperature,
            **generation_kwargs,
        )

        self.threshold = threshold

    def prompt(self, examples: str, explanation: str) -> list[dict]:
        return prompt(examples, explanation)

    def mean_n_activations_ceil(self, examples: list[ActivatingExample]):
        """
        Calculate the ceiling of the average number of activations in each example.
        """
        avg = sum(
            len(torch.nonzero(example.activations)) for example in examples
        ) / len(examples)

        return ceil(avg)

    def _prepare(self, record: LatentRecord) -> list[Sample]:
        """
        Prepare and shuffle a list of samples for classification.
        """
        assert len(record.test) > 0, "No test records found"

        n_incorrect = self.mean_n_activations_ceil(record.test)

        if len(record.not_active) > 0:
            samples = examples_to_samples(
                record.not_active,
                n_incorrect=n_incorrect,
                highlighted=True,
            )

        else:
            samples = []

        samples.extend(
            examples_to_samples(
                record.test,
                n_incorrect=0,
                highlighted=True,
            )
        )
        return samples
