from ... import det_config as CONFIG
import torch
import random
from typing import List

class Sample:

    def __init__(self, example, quantile, highlighted, activates):
        self.quantile = quantile
        self.highlighted = highlighted
        self.activates = activates
        self.marked = False

        self.text = self._prepare_example(
            example.str_toks,
            example.activations,
            n_incorrect=CONFIG.n_incorrect,
            threshold=CONFIG.threshold,
            highlight=highlighted
        )

    def default(self):
        return {
            "text": self.text,
            "quantile": self.quantile,
            "highlighted": self.highlighted,
            "activates": self.activates,
            "marked": self.marked
        }

    def _prepare_example(
        self,
        tokens: List[str], 
        activations,
        n_incorrect=0,
        threshold=0.0,
        highlight=False
    ) -> str:
        # Just join if not highlighting tokens
        if not highlight:
            return "".join(tokens)

        # Get all tokens below the activation threshold
        threshold = threshold * activations.max()
        below_threshold = torch.nonzero(
            activations <= threshold
        ).squeeze()

        # Randomly sample n_incorrect tokens below the threshold
        random.seed(CONFIG.seed)
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
                result.append("<<")

                while (
                    i < len(tokens) 
                    and check(i)
                ):
                    result.append(tokens[i])
                    i += 1

                result.append(">>")
            else:
                result.append(tokens[i])
                i += 1

        return "".join(result)