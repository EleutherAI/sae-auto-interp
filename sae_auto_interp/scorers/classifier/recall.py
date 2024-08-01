from typing import List

from transformers import PreTrainedTokenizer

from ...clients.client import Client
from ...features import FeatureRecord
from .classifier import Classifier
from .prompts.recall_prompt import prompt
from .sample import Sample, examples_to_samples


class RecallScorer(Classifier):
    name = "recall"

    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer,
        verbose: bool = False,
        batch_size: int = 10,
        log_prob: bool = False,
        **generation_kwargs,
    ):
        super().__init__(
            client=client,
            tokenizer=tokenizer,
            verbose=verbose,
            batch_size=batch_size,
            log_prob=log_prob,
            **generation_kwargs,
        )

        self.prompt = prompt

    def _prepare(self, record: FeatureRecord) -> List[List[Sample]]:
        """
        Prepare and shuffle a list of samples for classification.
        """

        samples = examples_to_samples(
            record.random_examples,
            distance=-1,
            ground_truth=False,
            tokenizer=self.tokenizer,
        )

        for i, examples in enumerate(record.test):
            samples.extend(
                examples_to_samples(
                    examples,
                    distance=i + 1,
                    ground_truth=True,
                    tokenizer=self.tokenizer,
                )
            )

        return samples
