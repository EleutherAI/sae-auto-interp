from dataclasses import dataclass
from typing import List

@dataclass
class Example:
    tokens: List[int]
    activations: List[float]
    
    def __hash__(self) -> int:
        return hash(tuple(self.tokens))

    def __eq__(self, other: 'Example') -> bool:
        return self.tokens == other.tokens
    
    @property
    def max_activation(self):
        return max(self.activations)
    
    @property
    def text(self):
        return "".join(self.str_toks)