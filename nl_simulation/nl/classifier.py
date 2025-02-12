import asyncio
import json
import random
import re
from abc import abstractmethod

import numpy as np
from transformers import PreTrainedTokenizer

from sae_auto_interp.clients.client import Client
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.logger import logger
from sae_auto_interp.scorers.scorer import Scorer, ScorerResult
from sae_auto_interp.scorers.classifier.sample import ClassifierOutput, Sample


class Classifier(Scorer):
    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer,
        verbose: bool,
        batch_size: int,
        log_prob: bool,
        contexts:bool = False,
        score:bool=False,
        **generation_kwargs,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.verbose = verbose

        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs
        self.log_prob = log_prob
        self.contexts = contexts
        self.score = score
        
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

    @abstractmethod
    def _prepare(self, record: FeatureRecord) -> list[list[Sample]]:
        pass

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
        if response is None:
            array = [-1] * self.batch_size
            conditional_probabilities = [-1] * self.batch_size
            probabilities = [-1] * self.batch_size
            
        else:
            selections = response.text
            logprobs = response.logprobs if self.log_prob else None
            try:
                array,conditional_probabilities, probabilities = self._parse(selections, logprobs)
            except Exception as e:
                logger.error(f"Parsing selections failed: {e}")
                array = [-1] * self.batch_size
                conditional_probabilities = [-1] * self.batch_size
                probabilities = [-1] * self.batch_size
    
        results = []
        correct = []
        response = []
        for i, sample in enumerate(batch):
            result = sample.data
            prediction = array[i] 
            result.prediction = prediction
            result.correct = prediction == result.ground_truth
            correct.append(result.ground_truth)
            response.append(prediction)
            if self.log_prob:
                result.probability = probabilities[i]
                result.conditional_probability = conditional_probabilities[i]
            results.append(result)

            if self.verbose:
                result.text = sample.text
        return results

    def _parse(self, string, logprobs=None):
        pattern = r"\[.*?\]"
        if self.finetuned:
           string = "["+string+"]"
        
        match = re.search(pattern, string)

        try:
            array = json.loads(match.group(0))
            assert len(array) == self.batch_size
            if self.log_prob:
                conditional_probabilities,probabilities = self._parse_logprobs(logprobs)
                assert len(conditional_probabilities) == self.batch_size
                return array, conditional_probabilities, probabilities
            conditional_probabilities = None
            probabilities = None
            return array, conditional_probabilities, probabilities
        except (json.JSONDecodeError, AssertionError, AttributeError) as e:
            logger.error(f"Parsing array failed: {e}")
            return [-1] * self.batch_size, [-1] * self.batch_size, [-1] * self.batch_size

    def _parse_logprobs(self, logprobs):
        #Logprobs will be a list of 5 probabilites for each token in the response
        # The response will be in the form of [x, x, x, ...] for each position we
        # need to get the probability of 1 or 0 
        conditional_probabilities = []
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
                    conditional_probabilities.append(prob_1/(prob_0+prob_1))
                    probabilities.append(prob_1)
                else:
                    conditional_probabilities.append(0)
                    probabilities.append(0)
        return conditional_probabilities,probabilities



    def _build_prompt(
        self,
        explanation: str,
        batch: list[Sample],
    ) -> str:
        """
        Prepare prompt for generation.
        """
        if self.finetuned:
            examples = "".join(sample.text for sample in batch)
        else:
            examples = "\n".join(
                f"{sample.text}" for i, sample in enumerate(batch)
        )
        return self.prompt(explanation=explanation, examples=examples, contexts=self.contexts,score=self.score)

    def _batch(self, samples):
        return [
            samples[i : i + self.batch_size]
            for i in range(0, len(samples), self.batch_size)
        ]
