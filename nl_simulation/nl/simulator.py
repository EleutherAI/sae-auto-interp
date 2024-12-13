from dataclasses import dataclass
import asyncio
import json
import random
import re
from abc import abstractmethod

import numpy as np
from transformers import PreTrainedTokenizer

from sae_auto_interp.clients.client import Client
from sae_auto_interp.features import FeatureRecord,Example
from sae_auto_interp.logger import logger
from sae_auto_interp.scorers.scorer import Scorer, ScorerResult
from sae_auto_interp.scorers.classifier.sample import  Sample,ClassifierOutput
from prompting import simulation_prompt
L = "<<"
R = ">>"


@dataclass
class SimulatorOutput:
    text: str
    """Text"""

    activation: float = 0
    """The activation of the example"""

    predicted_quantile: int = 0
    """The predicted quantiles of the example"""

    expected_quantile: float = 0
    """The expected quantiles of the example"""

    
class Simulator(Scorer):
    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer,
        verbose: bool = False,
        batch_size: int = 1,
        log_prob: bool = False,
        **generation_kwargs,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.verbose = verbose

        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs
        self.log_prob = log_prob

        self.prompt = simulation_prompt
    async def __call__(
        self,
        record: FeatureRecord,
    ) -> list[ClassifierOutput]:
        samples = self._prepare(record)
        random.shuffle(samples)
        samples = self._batch(samples)
        results = await self._query(
            record.explanation,
            samples,
        )
        return ScorerResult(record=record, score=results)

    def _prepare(self, record: FeatureRecord) -> list[list[Sample]]:
        """
        Prepare and shuffle a list of samples for classification.
        """

        samples = examples_to_samples(
                record.test[0],
                tokenizer=self.tokenizer,
            )
    
        return samples

    async def _query(
        self,
        explanation: str,
        batches: list[list[Sample]],
    ) -> list[Sample]:
        """
        Send and gather batches of samples to the model.
        """
        sem = asyncio.Semaphore(1)

        async def _process(explanation, batch):
            async with sem:
                result = await self._generate(explanation, batch)
                
                return result
    
        tasks = [asyncio.create_task(_process(explanation, batch)) for batch in batches]
        results = await asyncio.gather(*tasks)

        return sum(results, [])
    

    async def _generate(
        self, explanation: str, batch: list[Sample]
    ) -> list[ClassifierOutput]:
        """
        Generate predictions for a batch of samples.
        """

        prompt = self._build_prompt(explanation, batch)
        if self.log_prob:
            self.generation_kwargs["logprobs"] = True
            self.generation_kwargs["top_logprobs"] = 10
        response = await self.client.generate(prompt, **self.generation_kwargs)
        #print(prompt,response)
        if response is None:
            array = [-1] * self.batch_size
            probabilities = [-1] * self.batch_size
            
        else:
            selections = response.text
            logprobs = response.logprobs if self.log_prob else None
            try:
                array, probabilities = self._parse(selections, logprobs)
            except Exception as e:
                logger.error(f"Parsing selections failed: {e}")
                array = [-1] * self.batch_size
                probabilities = [-1] * self.batch_size
    
        results = []
        for i, sample in enumerate(batch):
            result = sample.data
            prediction = array[i] 
            result.predicted_quantile = prediction
            if self.log_prob:
                result.expected_quantile = probabilities[i]
            results.append(result)

            if self.verbose:
                result.text = sample.text
        return results

    def _parse(self, string, logprobs=None):
        pattern = r"\[.*?\]"
        
        match = re.search(pattern, string)
        try:
            array = json.loads(match.group(0))
            assert len(array) == self.batch_size
            if self.log_prob:
                probabilities = self._parse_logprobs(logprobs)
                assert len(probabilities) == self.batch_size
                return array, probabilities
            probabilities = None
            return array, probabilities
        except (json.JSONDecodeError, AssertionError, AttributeError) as e:
            logger.error(f"Parsing array failed: {e}")
            return [-1] * self.batch_size, [-1] * self.batch_size

    def _parse_logprobs(self, logprobs):
        #Logprobs will be a list of 10 probabilites for each token in the response
        # The response will be in the form of [x, x, x, ...] for each position we
        # need to get the probability of each of the 10 integer and compute the expected value
        probabilities = []
        
        for i in range(len(logprobs)):
            if logprobs[i].token in ["0","1","2","3","4","5","6","7","8","9"]:
                cumulative_number = 0
                top_logprobs = logprobs[i].top_logprobs
                for i in range(len(top_logprobs)):
                    token = top_logprobs[i].token
                    logprob = top_logprobs[i].logprob
                    probability = np.exp(logprob).item()
                    try:
                        integer = int(token)
                        cumulative_number += probability*integer
                    except:
                        pass
                probabilities.append(cumulative_number)
        return probabilities



    def _build_prompt(
        self,
        explanation: str,
        batch: list[Sample],
    ) -> str:
        """
        Prepare prompt for generation.
        """
        
        examples ="Example 1: "+"".join(
                f"{sample.text}" for i, sample in enumerate(batch)
        )
        return self.prompt(explanation=explanation, examples=examples)

    def _batch(self, samples):
        return [
            samples[i : i + self.batch_size]
            for i in range(0, len(samples), self.batch_size)
        ]

def examples_to_samples(
    example: Example,
    tokenizer: PreTrainedTokenizer,
) -> Sample:
    
    text,clean = highlight_last_token(example, tokenizer)
    activation = example.activations[-1]
    sample = Sample(
                text=text,
                data=SimulatorOutput(
                    text=clean,
                    activation=activation
            ),
        )
    return [sample]

def highlight_last_token(example: Example, tokenizer: PreTrainedTokenizer) -> str:
    tokens = example.tokens
    decoded_tokens = [tokenizer.decode([token]) for token in tokens]
    clean = "".join(decoded_tokens)
    highlighted = ""
    for i, token in enumerate(decoded_tokens):
        if i == len(tokens) - 1:
            highlighted += L + token + R
        else:
            highlighted += token

    return highlighted,clean
