from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ...clients.client import Client
from ...features import FeatureRecord, Example
from .classifier import Classifier
from .prompts.detection_prompt import prompt
from .sample import Sample, examples_to_samples


class DetectionScorer(Classifier):
    name = "detection"

    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
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

    def _prepare(self, record: FeatureRecord) -> list[list[Sample]]:
        """
        Prepare and shuffle a list of samples for classification.
        """

        # check if random_examples is a list of lists or a list of examples
        if isinstance(record.random_examples[0], tuple):
            # Here we are using neighbours
            samples = []
            for i, (examples, neighbour) in enumerate(record.random_examples):
                samples.extend(
                    examples_to_samples(
                        examples,
                        distance=-neighbour.distance,
                        ground_truth=False,
                        tokenizer=self.tokenizer,
                    )
                )
        elif isinstance(record.random_examples[0], Example):
            # This is if we dont use neighbours
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
