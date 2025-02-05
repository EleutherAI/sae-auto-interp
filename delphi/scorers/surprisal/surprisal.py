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
from torch.nn.functional import cross_entropy
from transformers import PreTrainedTokenizer

from ...clients.client import Client
from ...features import Example, FeatureRecord
from ..scorer import Scorer, ScorerResult
from .prompts import BASEPROMPT as base_prompt


@dataclass
class SurprisalOutput:
    text: str
    """The text that was used to evaluate the surprisal"""

    distance: float | int
    """Quantile or neighbor distance"""

    no_explanation: list[float] = 0
    """What is the surprisal of the model with no explanation"""

    explanation: list[float] = 0
    """What is the surprisal of the model with an explanation"""

    activations: list[float] = 0
    """What are the activations of the model"""

class Sample(NamedTuple):
    text: str
    activations: list[float]
    data: SurprisalOutput


class SurprisalScorer(Scorer):
    name = "surprisal"

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
    ) -> list[SurprisalOutput]:
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

    def compute_loss_with_kv_cache(self, explanation:str, samples: list[Sample], batch_size=2):
        
        
        #print(explanation_prompt)
        model = self.model
        tokenizer = self.model.tokenizer
        # Tokenize explanation
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        explanation_tokens = tokenizer.encode(explanation, return_tensors="pt", add_special_tokens=False).to(model.device)
        # Generate KV cache for explanation
        explanation_tokens = explanation_tokens.repeat_interleave(batch_size, dim=0)
        
        with torch.inference_mode():
            outputs = model(input_ids=explanation_tokens, use_cache=True)
            kv_cache = outputs.past_key_values
        total_losses = []
        for i in range(0,len(samples),batch_size):
            batch_samples = samples[i:i+batch_size]
            current_batch_size = len(batch_samples)
            if current_batch_size < batch_size:
                explanation_tokens = explanation_tokens.repeat_interleave(current_batch_size, dim=0)
                with torch.inference_mode():
                    outputs = model(input_ids=explanation_tokens, use_cache=True)
                    kv_cache = outputs.past_key_values
                
            # Tokenize full input (explanation + prompts)
            full_inputs = [sample.text for sample in batch_samples]
            tokenized_inputs = tokenizer(full_inputs, return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
            #print(tokenized_inputs.input_ids.shape)
            
            # Prepare input for the model (including explanation)
            input_ids = tokenized_inputs.input_ids
            attention_mask = tokenized_inputs.attention_mask
            labels = input_ids.clone()
            labels[~attention_mask.bool()] = -100
            # Forward pass using KV cache
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, 
                                #attention_mask=attention_mask, 
                                past_key_values=kv_cache,
                                )
            # Compute loss
            logits = outputs.logits
            
            for j, logit in enumerate(logits):
                
                loss = cross_entropy(logit[:-1], labels[j][1:],reduction="none").tolist()
                #Remove the trailing zeros from the loss, by looking at the attention mask
                loss = loss[:attention_mask[j].sum().item()]

                #print(len(loss))
                total_losses.append(loss)
        return total_losses



    def _query(self, explanation: str, samples: list[Sample]) -> list[SurprisalOutput]:

        explanation_prompt = base_prompt + "Description: \n" + explanation + "\n Sentences:\n"
        no_explanation_prompt = base_prompt + "Description: \n" + "Various unrelated sentences." + "\n Sentences:\n"
        


        no_explanation_losses = self.compute_loss_with_kv_cache(no_explanation_prompt, samples,batch_size=10)
        explanation_losses = self.compute_loss_with_kv_cache(explanation_prompt, samples,batch_size=10)
        results = []
        for i in range(len(samples)):
            #print(i)
            samples[i].data.no_explanation = no_explanation_losses[i]
            samples[i].data.explanation = explanation_losses[i]
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
                data=SurprisalOutput(
                    activations=activations,
                    text=text,
                    **sample_kwargs
                ),
            )
        )

    return samples
    
