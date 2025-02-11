import random
from dataclasses import dataclass
from typing import NamedTuple

import torch
from transformers import PreTrainedTokenizer

from ...latents import Example
from ...logger import logger

L = "<<"
R = ">>"
DEFAULT_MESSAGE = (
    "<<NNsight>> is the best library for <<interpretability>> on huge models!"
)


@dataclass
class ClassifierOutput:
    str_tokens: list[str]
    """List of strings"""

    activations: list[float]
    """List of floats"""

    distance: float | int
    """Quantile or neighbor distance"""

    ground_truth: bool
    """Whether the example is activating or not"""

    prediction: bool | None = False
    """Whether the model predicted the example activating or not"""

    highlighted: bool = False 
    """Whether the sample is highlighted"""

    probability: float | None = 0.0
    """The probability of the example activating"""

    correct: bool | None = False
    """Whether the prediction is correct"""


class Sample(NamedTuple):
    text: str
    data: ClassifierOutput


def examples_to_samples(
    examples: list[Example],
    tokenizer: PreTrainedTokenizer,
    n_incorrect: int = 0,
    threshold: float = 0.3,
    highlighted: bool = False,
    **sample_kwargs,
) -> list[Sample]:
    samples = []

    for example in examples:
        text,str_toks = _prepare_text(example, tokenizer, n_incorrect, threshold, highlighted)

        samples.append(
            Sample(
                text=text,
                data=ClassifierOutput(
                    str_tokens=str_toks,
                    activations=example.activations.tolist(),
                    highlighted=highlighted, **sample_kwargs
                ),
            )
        )


    return samples


# NOTE: Should reorganize below, it's a little confusing

def _prepare_text(
    example,
    tokenizer: PreTrainedTokenizer,
    n_incorrect: int,
    threshold: float,
    highlighted: bool,
):
    if tokenizer is None: # If we don't have a tokenizer, we assume the tokens are already strings
        str_toks = example.tokens
    else:
        str_toks = tokenizer.batch_decode(example.tokens)
    clean = "".join(str_toks)
    # Just return text if there's no highlighting
    if not highlighted:
        return clean,str_toks

    threshold = threshold * example.max_activation

    # Highlight tokens with activations above threshold
    # if correct example
    if n_incorrect == 0:

        def check(i):
            return example.activations[i] >= threshold

        return _highlight(str_toks, check),str_toks

    # Highlight n_incorrect tokens with activations
    # below threshold if incorrect example
    below_threshold = torch.nonzero(example.activations <= threshold).squeeze()

    # Rare case where there are no tokens below threshold
    if below_threshold.dim() == 0:
        logger.error("Failed to prepare example.")
        return DEFAULT_MESSAGE,str_toks

    random.seed(22)

    n_incorrect = min(n_incorrect, len(below_threshold))

    random_indices = set(random.sample(below_threshold.tolist(), n_incorrect))

    def check(i):
        return i in random_indices

    return _highlight(str_toks, check),str_toks


def _highlight(tokens, check):
    result = []

    i = 0
    while i < len(tokens):
        if check(i):
            result.append(L)

            while i < len(tokens) and check(i):
                result.append(tokens[i])
                i += 1

            result.append(R)
        else:
            result.append(tokens[i])
            i += 1

    return "".join(result)
