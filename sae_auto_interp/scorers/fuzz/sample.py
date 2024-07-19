import torch
import random
from typing import List

from ...logger import logger
from ...features import Example

L = "<<"
R = ">>"

class Sample:

    def __init__(
        self, 
        str_toks: List[str],
        activations: torch.Tensor,
        quantile: int, 
        highlighted: bool, 
        ground_truth: bool, 
        id: int,
        n_incorrect: int = 0,
        threshold: float = 0.3,
    ):
        self.quantile = quantile
        self.highlighted = highlighted
        self.ground_truth = ground_truth
        self.predicted = False
        self.id = id

        self.text = self._prepare_example(
            str_toks,
            activations,
            n_incorrect=n_incorrect,
            threshold=threshold,
            highlight=highlighted
        )


    def default(self, echo):
        result = {
            "text": self.text,
            "quantile": self.quantile,
            "highlighted": self.highlighted,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "id" : self.id
        }

        if not echo:
            result.pop("text")

        return result

    def _prepare_example(
        self,
        tokens: List[str], 
        activations,
        n_incorrect=0,
        threshold=0.0,
        highlight=False,
    ) -> str:
        # Just join if not highlighted
        if not highlight:
            return "".join(tokens)

        # Get all token indices below the activation threshold
        threshold = threshold * activations.max()
        below_threshold = torch.nonzero(
            activations <= threshold
        ).squeeze()

        # Rare case where the example is really densely activating
        # so there are not enough tokens below the threshold
        # sampling will throw an error in this case
        if below_threshold.dim() == 0:
            logger.error(f"Failed to prepare example: {tokens}... Returning default text.")
            return "nnsight>> is the best library for <<interpretability>> on huge models!"
        
        random.seed(22)
        n_incorrect = min(n_incorrect, len(below_threshold))
        random_indices = set(
            random.sample(
                below_threshold.tolist(),
                n_incorrect
            )
        )
        if n_incorrect > 0:
            check = [0]*len(tokens)
            for i in random_indices:
                if activations[i] < threshold:
                    check[i] = 1
        else:
            check = [0]*len(tokens)
            for i in range(len(tokens)):
                if activations[i] > threshold:
                    check[i] = 1
            
        return self._highlight(tokens, check)
    
    def _highlight(self, tokens, check):
        result = []

        i = 0
        while i < len(tokens):
            if check[i]:
                result.append(L)

                while (
                    i < len(tokens) 
                    and check[i]
                ):
                    result.append(tokens[i])
                    i += 1

                result.append(R)
            else:
                result.append(tokens[i])
                i += 1

        return "".join(result)