import asyncio
from typing import List
import random 

from transformers import PreTrainedTokenizer

from .classifier import Classifier
from .sample import examples_to_samples, Sample
from .prompts.recall_prompt import prompt
from ...clients.client import Client
from ...features import FeatureRecord

class NeighborScorer(Classifier):
    name = "neighbor"

    def __init__(
        self, 
        client: Client, 
        tokenizer: PreTrainedTokenizer,
        verbose: bool = False,
        batch_size: int = 5,
        **generation_kwargs
    ):
        super().__init__(
            client = client,
            tokenizer = tokenizer,
            verbose=verbose,
            batch_size = batch_size,
            **generation_kwargs
        )

        self.prompt = prompt

    def _prepare(
        self, 
        record: FeatureRecord
    ) -> List[List[Sample]]:
        """
        Prepare and shuffle a list of samples for classification.
        """

        samples = examples_to_samples(
            record.test[0],
            distance = 0,
            ground_truth = True,
            tokenizer = self.tokenizer
        )

        for distance, neighbor in record.neighbors.items():

            if neighbor is None:
                continue

            samples.extend(
                examples_to_samples(
                    neighbor.examples[:10],
                    distance = distance,
                    ground_truth = False,
                    tokenizer = self.tokenizer
                )
            )

        return samples