from dataclasses import dataclass

import blobfile as bf
import orjson
from torchtyping import TensorType
from transformers import AutoTokenizer


@dataclass
class Example:
    tokens: TensorType["seq"]
    activations: TensorType["seq"]
    normalized_activations: TensorType["seq"]

    def __hash__(self) -> int:
        return hash(tuple(self.tokens.tolist()))

    def __eq__(self, other: "Example") -> bool:
        return self.tokens.tolist() == other.tokens.tolist()

    @property
    def max_activation(self):
        return max(self.activations)


def prepare_examples(tokens, activations,max_activation):
    return [
        Example(
            tokens=toks,
            activations=acts,
            normalized_activations=(acts*10 / max_activation).floor(),
        )
        for toks, acts in zip(tokens, activations)
    ]


@dataclass
class Feature:
    module_name: int
    feature_index: int

    def __repr__(self) -> str:
        return f"{self.module_name}_feature{self.feature_index}"


class FeatureRecord:
    def __init__(
        self,
        feature: Feature,
    ):
        self.feature = feature

    @property
    def max_activation(self):
        return self.examples[0].max_activation

    def save(self, directory: str, save_examples=False):
        path = f"{directory}/{self.feature}.json"
        serializable = self.__dict__

        if not save_examples:
            serializable.pop("examples")
            serializable.pop("train")
            serializable.pop("test")

        serializable.pop("feature")
        with bf.BlobFile(path, "wb") as f:
            f.write(orjson.dumps(serializable))
    

    def display(
        self,
        tokenizer: AutoTokenizer,
        threshold: float = 0.0,
        n: int = 10,
    ) -> str:
        from IPython.core.display import HTML, display

        def _to_string(tokens: TensorType["seq"], activations: TensorType["seq"]) -> str:
            result = []
            i = 0

            max_act = max(activations)
            _threshold = max_act * threshold

            while i < len(tokens):
                if activations[i] > _threshold:
                    result.append("<mark>")
                    while i < len(tokens) and activations[i] > _threshold:
                        result.append(tokens[i])
                        i += 1
                    result.append("</mark>")
                else:
                    result.append(tokens[i])
                    i += 1
                return "".join(result)
        
        strings = [
            _to_string(tokenizer.batch_decode(example.tokens), example.activations)
            for example in self.examples[:n]
        ]

        display(HTML("<br><br>".join(strings)))
