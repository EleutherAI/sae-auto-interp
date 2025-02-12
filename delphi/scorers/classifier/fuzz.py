from math import ceil
from typing import List

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ...clients.client import Client
from ...latents.latents import Example
from ...latents import LatentRecord
from ..scorer import Scorer
from .classifier import Classifier
from .prompts.fuzz_prompt import prompt
from .sample import Sample, examples_to_samples


class FuzzingScorer(Classifier, Scorer):
    name = "fuzz"

    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        verbose: bool = False,
        n_examples_shown: int = 10,
        threshold: float = 0.3,
        log_prob: bool = False,
        temperature: float = 0.,
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
                        it harder for models to generate anwers in the correct format
            log_prob: Whether to use log probabilities to allow for AUC calculation
            generation_kwargs: Additional generation kwargs
        """
        super().__init__(
            client=client,
            tokenizer=tokenizer,
            verbose=verbose,
            n_examples_shown=n_examples_shown,
            log_prob=log_prob,
            temperature=temperature,
            **generation_kwargs,
        )

        self.threshold = threshold
        self.prompt = prompt

    def mean_n_activations_ceil(self, examples: list[Example]):
        """
        Calculate the ceiling of the average number of activations in each example.
        """
        avg = sum(
            len(torch.nonzero(example.activations)) for example in examples
        ) / len(examples)

        return ceil(avg)

    def _prepare(self, record: LatentRecord) -> list[list[Sample]]:
        """
        Prepare and shuffle a list of samples for classification.
        """
        assert (
            len(record.test) > 0 
            and len(record.test[0]) > 0
        ), "No test records found"

        defaults = {
            "highlighted": True,
            "tokenizer": self.tokenizer,
        }
        all_examples = []
        for examples_chunk in record.test:
            all_examples.extend(examples_chunk)

        n_incorrect = self.mean_n_activations_ceil(all_examples)
        if len(record.not_active) > 0:
            if isinstance(record.not_active[0], list):
                # Here we are using neighbours
                samples = []
                for i, examples in enumerate(record.not_active):
                    samples.extend(
                        examples_to_samples(
                            examples,
                            distance=-record.neighbours[i].distance,
                            ground_truth=False,
                            n_incorrect=n_incorrect,
                            **defaults,
                        )
                    )
            elif isinstance(record.not_active[0], Example):
                # This is if we dont use neighbours
                samples = examples_to_samples(
                    record.not_active,
                    distance=-1,
                    ground_truth=False,
                    n_incorrect=n_incorrect,
                    **defaults,
                )
        else:
            samples = []

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
