from dataclasses import dataclass
from typing import List
from torchtyping import TensorType

@dataclass
class Example:
    tokens: TensorType["seq"]
    activations: TensorType["seq"]
    
    def __hash__(self) -> int:
        return hash(tuple(self.tokens.tolist()))

    def __eq__(self, other: 'Example') -> bool:
        return self.tokens == other.tokens
    
    @property
    def max_activation(self):
        return max(self.activations)
    
    @property
    def text(self):
        return "".join(self.str_toks)

    @staticmethod
    def prepare_examples(tokens, activations):
        return [
            Example(
                tokens=toks,
                activations=acts,
            )
            for toks, acts in zip(
                tokens, 
                activations
            )
        ]