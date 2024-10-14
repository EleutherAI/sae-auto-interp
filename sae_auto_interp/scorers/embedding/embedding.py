import asyncio
import json
import random
import re
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, NamedTuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer
from ...clients.client import Client
from ...features import Example, FeatureRecord
from ..scorer import Scorer, ScorerResult


@dataclass
class EmbedingOutput:
    text: str
    """The text that was used to evaluate the similarity"""

    distance: float | int
    """Quantile or neighbor distance"""

    similarity: list[float] = 0
    """What is the similarity of the example to the explanation"""


class Sample(NamedTuple):
    text: str
    activations: list[float]
    data: EmbedingOutput


class EmbedingScorer(Scorer):
    name = "embeding"

    def __init__(
        self,
        model,
        tokenizer,
        verbose: bool,
        batch_size: int,
        **generation_kwargs,
    ):
        self.model = model
        self.verbose = verbose
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs
        
    async def __call__(
        self,
        record: FeatureRecord,
    ) -> list[EmbedingOutput]:
        samples = self._prepare(record)

        random.shuffle(samples)
        results = self._query(
            record.explanation,
            samples,
        )
        
        return ScorerResult(record=record, score=results)

    def _prepare(self, record: FeatureRecord) -> list[list[Sample]]:
        """
        Prepare and shuffle a list of samples for classification.
        """

        defaults = {
            "tokenizer": self.tokenizer,
        }
        samples = examples_to_samples(
            record.extra_examples,
            distance=-1,
            **defaults,
        )

        for i, examples in enumerate(record.test):
            samples.extend(
                examples_to_samples(
                    examples,
                    distance=i + 1,
                    **defaults,
                )
            )

        return samples



    def _query(self, explanation: str, samples: list[Sample]) -> list[EmbedingOutput]:

        explanation_prompt = "Instruct: Retrieve sentences that could be related to the explanation.\nQuery:" + explanation 
        query_embeding = self.model.encode(explanation_prompt)
        
        samples_text = [sample.text for sample in samples]
    
        # # Temporary batching
        # sample_embedings = []
        # for i in range(0, len(samples_text), 10):
        #     sample_embedings.extend(self.model.encode(samples_text[i:i+10]))
        sample_embedings = self.model.encode(samples_text)
        similarity = self.model.similarity(query_embeding,sample_embedings)[0]
        
        results = []
        for i in range(len(samples)):
            #print(i)
            samples[i].data.similarity = similarity[i].item()
            results.append(samples[i].data)
        return results
        



def examples_to_samples(
    examples: list[Example],
    tokenizer: PreTrainedTokenizer,
    **sample_kwargs,
) -> list[Sample]:
    samples = []    
    for example in examples:
        text = "".join(tokenizer.batch_decode(example.tokens))
        activations = example.activations.tolist()
        samples.append(
            Sample(
                text=text,
                activations=activations,
                data=EmbedingOutput(
                    text=text,
                    **sample_kwargs
                ),
            )
        )

    return samples
    
