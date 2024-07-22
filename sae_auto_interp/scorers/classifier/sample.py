import torch
import random
from typing import List, NamedTuple

from ...logger import logger
from ...features import Example

from dataclasses import dataclass
from transformers import PreTrainedTokenizer


from torchtyping import TensorType

L = "<<"
R = ">>"

@dataclass
class ClassifierOutput:

    id: int
    """Hashed tokens"""

    distance: float | int
    """Quantile or neighbor distance"""

    ground_truth: bool
    """Whether the example is correct or not"""

    predicted: bool = False
    """Whether the model predicted the example correctly"""

    highlighted: bool = False
    """Whether the sample is highlighted"""


class Sample(NamedTuple):

    text: str

    data: ClassifierOutput


def examples_to_samples(
    examples: List[Example],
    tokenizer: PreTrainedTokenizer,
    n_incorrect: int = 0,
    threshold: float = 0.0,
    **sample_kwargs
) -> List[Sample]:

    samples = []

    for example in examples:

        samples.append(
            Sample(
                text = tokenizer.batch_decode(example.tokens),
                data = ClassifierOutput(
                    id = hash(example),
                    **sample_kwargs
                )
            )
        )

    return samples


# def _highlight(
#     self,
#     example: Example,
#     n_incorrect: int = 0,
#     threshold: float = 0.0,
#     **sample_kwargs
# ) -> str:
#     threshold = threshold * example.max_activation

#     below_threshold = torch.nonzero(
#         example.activations <= threshold
#     ).squeeze()

#     random.seed(22)

#     n_incorrect = min(n_incorrect, len(below_threshold))

#     random_indices = set(
#         random.sample(
#             below_threshold.tolist(),
#             n_incorrect
#         )
#     )
        
#     return self._highlight(tokens, check)

# def _highlight(self, tokens, check):
#     result = []

#     i = 0
#     while i < len(tokens):
#         if check[i]:
#             result.append(L)

#             while (
#                 i < len(tokens) 
#                 and check[i]
#             ):
#                 result.append(tokens[i])
#                 i += 1

#             result.append(R)
#         else:
#             result.append(tokens[i])
#             i += 1

#     return "".join(result)