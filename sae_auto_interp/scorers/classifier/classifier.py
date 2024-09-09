import asyncio
import json
import random
import re
from abc import abstractmethod
from typing import List

import numpy as np
from transformers import PreTrainedTokenizer

from ...clients.client import Client
from ...features import FeatureRecord
from ..scorer import Scorer, ScorerResult
from .sample import ClassifierOutput, Sample


class Classifier(Scorer):
    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer,
        verbose: bool,
        batch_size: int,
        log_prob: bool,
        **generation_kwargs,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.verbose = verbose

        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs
        self.log_prob = log_prob



    async def __call__(
        self,
        record: FeatureRecord,
    ) -> List[ClassifierOutput]:
        samples = self._prepare(record)

        random.shuffle(samples)

        results = await self._query(
            record.explanation,
            self._batch(samples),
        )

        return ScorerResult(record=record, score=results)

    @abstractmethod
    def _prepare(self, record: FeatureRecord) -> List[List[Sample]]:
        pass

    async def _query(
        self,
        explanation: str,
        batches: List[List[Sample]],
    ) -> List[Sample]:
        """
        Send and gather batches of samples to the model.
        """

        tasks = [self._generate(explanation, batch) for batch in batches]

        results = await asyncio.gather(*tasks)

        return sum(results, [])

    async def _generate(
        self, explanation: str, batch: List[Sample]
    ) -> List[ClassifierOutput]:
        """
        Generate predictions for a batch of samples.
        """

        prompt = self._build_prompt(explanation, batch)
        if self.log_prob:
            self.generation_kwargs["logprobs"] = True
            self.generation_kwargs["top_logprobs"] = 5
            #self.generation_kwargs["echo"] = True
            #self.generation_kwargs["stop"] = "]"
            response = await self.client.generate(prompt, **self.generation_kwargs,raw=True)
            if response is None:
                array = [-1] * self.batch_size
                probabilities = [-1] * self.batch_size
            else:
                selections = response.choices[0].message.content
                logprobs = response.choices[0].logprobs.content
                array, probabilities = self._parse(selections, logprobs)
        else:
            selections = await self.client.generate(prompt, **self.generation_kwargs)
            if selections is None:
                array = [-1] * self.batch_size
            else:
                array = self._parse(selections)

        
        results = []

        for i, sample in enumerate(batch):
            result = sample.data
            prediction = array[i] 
            result.prediction = prediction
            result.correct = prediction == result.ground_truth
            if self.log_prob:
                result.probability = probabilities[i]
            results.append(result)

            if self.verbose:
                result.text = sample.text

        return results

    def _parse(self, string,logprobs=None):
        pattern = r"\[.*?\]"
        match = re.search(pattern, string)

        try:
            array = json.loads(match.group(0))
            assert len(array) == self.batch_size
            if logprobs:
                probabilities = self._parse_logprobs(logprobs)
                assert len(probabilities) == self.batch_size
                return array, probabilities
            return array
        except (json.JSONDecodeError, AssertionError, AttributeError):
            if logprobs:
                return [-1] * self.batch_size, [-1] * self.batch_size
            return [-1] * self.batch_size

    def _parse_logprobs(self, logprobs):
        #Logprobs will be a list of 5 probabilites for each token in the response
        # The response will be in the form of [x, x, x, ...] for each position we
        # need to get the probability of 1 or 0 
        probabilities = []
        
        for i in range(len(logprobs)):
            if "1" in logprobs[i].token or "0" in logprobs[i].token:
                top_logprobs = logprobs[i].top_logprobs
                prob_0 = 0
                prob_1 = 0
                for i in range(len(top_logprobs)):
                    token = top_logprobs[i].token
                    logprob = top_logprobs[i].logprob
                    if "0" in token:
                        prob_0 += np.exp(logprob).item()
                    elif "1" in token:
                        prob_1 += np.exp(logprob).item()
                if prob_0+prob_1>0:
                    probabilities.append(prob_1/(prob_0+prob_1))
                else:
                    probabilities.append(0)
        return probabilities




    def _build_prompt(
        self,
        explanation: str,
        batch: List[Sample],
    ) -> str:
        """
        Prepare prompt for generation.
        """

        examples = "\n".join(
            f"Example {i}: {sample.text}" for i, sample in enumerate(batch)
        )

        return self.prompt(explanation=explanation, examples=examples)

    def _batch(self, samples):
        return [
            samples[i : i + self.batch_size]
            for i in range(0, len(samples), self.batch_size)
        ]
