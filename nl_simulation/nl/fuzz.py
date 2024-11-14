from dataclasses import dataclass
from math import ceil
from typing import List

import torch
from transformers import PreTrainedTokenizer

from sae_auto_interp.clients.client import Client
from sae_auto_interp.features import Example, FeatureRecord
from sae_auto_interp.logger import logger
from sae_auto_interp.scorers import Scorer

from sae_auto_interp.scorers.classifier.sample import Sample

from prompting import prompt,finetuned_prompt
from classifier import Classifier
L = "<<"
R = ">>"
DEFAULT_MESSAGE = (
    "<<NNsight>> is the best library for <<interpretability>> on huge models!"
)


@dataclass
class ClassifierOutput:
    text: str
    """Text"""

    ground_truth: bool
    """Whether the example is correct or not"""

    prediction: bool = False
    """Whether the model predicted the example correctly"""

    probability: float = 0.0
    """The probability of the prediction"""

class FuzzingScorer(Classifier):
    name = "fuzz"

    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer,
        log_prob:bool = True,
        finetuned:bool = False,
        contexts:bool = False,
        score:bool=False,
        **generation_kwargs,
    ):
        super().__init__(
            client=client,
            tokenizer=tokenizer,
            verbose=False,
            batch_size=1,
            log_prob=log_prob,
            contexts=contexts,
            score=score,
            **generation_kwargs,
        )
        self.prompt = finetuned_prompt if finetuned else prompt
        self.finetuned = finetuned
        self.contexts = contexts
        self.score = score
        
    def _prepare(self, record: FeatureRecord) -> list[list[Sample]]:
        """
        Prepare and shuffle a list of samples for classification.
        """

        samples = []
        if len(record.test) > 0:
            samples.extend(examples_to_samples(
                record.test[0],
                ground_truth=True,
                tokenizer=self.tokenizer,
                finetuned=self.finetuned,
            ))
        elif len(record.extra_examples) > 0:
            samples.extend(examples_to_samples(
                record.extra_examples[0],
                ground_truth=False,
                tokenizer=self.tokenizer,
                finetuned=self.finetuned,
            ))
        if self.contexts:
            explanation = ""
            for example in record.train:
                explanation += _highlight(example,self.tokenizer) + "\n"
            
            record.explanation = explanation
        return samples
def examples_to_samples(
    example: Example,
    tokenizer: PreTrainedTokenizer,
    ground_truth: bool,
    finetuned:bool = False,
) -> Sample:
    
    text,clean = highlight_last_token(example, tokenizer)
    if finetuned:
        text = clean
    sample = Sample(
                text=text,
                data=ClassifierOutput(
                    text=clean,
                    ground_truth=ground_truth
            ),
        )
    return [sample]

def _highlight(example,tokenizer):
    result = []
    tokens = tokenizer.batch_decode(example.tokens)
    activations = example.activations
    max_activation = example.max_activation
    i = 0
    while i < len(tokens):
        if activations[i] >= 0.7 * max_activation:
            result.append(L)

            while i < len(tokens) and activations[i] >= 0.7 * max_activation:
                result.append(tokens[i])
                i += 1

            result.append(R)
        else:
            result.append(tokens[i])
            i += 1

    return "".join(result)

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

