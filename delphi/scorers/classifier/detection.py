from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ...clients.client import Client
from ...latents import LatentRecord
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
        n_examples_shown: int = 10,
        log_prob: bool = False,
        temperature: float = 0.0,
        **generation_kwargs,
    ):
        """
        Initialize a DetectionScorer.

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

        self.prompt = prompt

    def _prepare(self, record: LatentRecord) -> list[list[Sample]]:
        """
        Prepare and shuffle a list of samples for classification.
        """

        samples = examples_to_samples(
            record.not_active,
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
