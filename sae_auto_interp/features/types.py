from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Feature:
    layer_index: int
    feature_index: int
    
    def __repr__(self) -> str:
        return f"layer{self.layer_index}_feature{self.feature_index}"
    
@dataclass
class Example:
    tokens: List[int]
    activations: List[float]

    def __hash__(self) -> int:
        if self.str_toks is None:
            raise ValueError("Cannot hash examples without decoding.")
        
        return hash(tuple(self.tokens))

    def __eq__(self, other) -> bool:
        if self.str_toks is None:
            raise ValueError("Cannot compare examples without decoding.")
        
        return self.tokens == other.tokens
    
    def decode(self, tokenizer):
        if tokenizer is None:
            self.str_toks = None
        self.str_toks = tokenizer.batch_decode(self.tokens)

    @property
    def max_activation(self):
        return max(self.activations)
    
    @property
    def text(self):
        return "".join(self.str_toks)