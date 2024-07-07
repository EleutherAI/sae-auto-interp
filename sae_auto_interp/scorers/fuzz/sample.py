import torch
import random
from typing import List

from ...logger import logger
from ... import det_config as CONFIG

class Sample:

    def __init__(self, example, quantile, highlighted, ground_truth, id, n_incorrect=0, echo=False):
        self.quantile = quantile
        self.highlighted = highlighted
        self.ground_truth = ground_truth
        self.predicted = False
        self.id = id
        self.echo = echo

        self.text = self._prepare_example(
            example.str_toks,
            example.activations,
            n_incorrect=n_incorrect,
            threshold=CONFIG.threshold,
            highlight=highlighted
        )


    def default(self):
        result = {
            "text": self.text,
            "quantile": self.quantile,
            "highlighted": self.highlighted,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "id" : self.id
        }

        if not self.echo:
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
        # Just join if not highlighting tokens
        if not highlight:
            return "".join(tokens)

        # Get all tokens below the activation threshold
        threshold = threshold * activations.max()
        below_threshold = torch.nonzero(
            activations <= threshold
        ).squeeze()

        # Rare case where the example is really densely activating
        # so there are not enough tokens below the threshold
        # sampling will throw an error in this case
        if below_threshold.dim() == 0:
            logger.error(f"Failed to prepare example: {tokens}")
            return "nnsight>> is the best library for <<interpretability>> on huge models!"
        
        random.seed(CONFIG.seed)
        n_incorrect = min(n_incorrect, len(below_threshold))
        random_indices = set(
            random.sample(
                below_threshold.tolist(),
                n_incorrect
            )
        )

        check = lambda i: activations[i] > threshold \
            or i in random_indices
        
        return self._highlight(tokens, check)
    
    def _highlight(self, tokens, check):
        result = []

        i = 0
        while i < len(tokens):
            if check(i):
                result.append(CONFIG.l)

                while (
                    i < len(tokens) 
                    and check(i)
                ):
                    result.append(tokens[i])
                    i += 1

                result.append(CONFIG.r)
            else:
                result.append(tokens[i])
                i += 1

        return "".join(result)