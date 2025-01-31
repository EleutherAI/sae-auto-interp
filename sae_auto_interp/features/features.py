from dataclasses import dataclass
from typing import Optional
import blobfile as bf
import orjson
from torchtyping import TensorType
from transformers import AutoTokenizer


@dataclass
class Example:
    """
    A single example of feature data.

    Attributes:
        tokens (TensorType["seq"]): Tokenized input sequence.
        activations (TensorType["seq"]): Activation values for the input sequence.
        normalized_activations (TensorType["seq"]): Normalized activation values.
    """
    tokens: TensorType["seq"]
    activations: TensorType["seq"]
    normalized_activations: Optional[TensorType["seq"]] = None
    
    @property
    def max_activation(self):
        """
        Get the maximum activation value.

        Returns:
            float: The maximum activation value.
        """
        return self.activations.max()


def prepare_examples(tokens, activations):
    """
    Prepare a list of examples from input tokens and activations.

    Args:
        tokens (List[TensorType["seq"]]): Tokenized input sequences.
        activations (List[TensorType["seq"]]): Activation values for the input sequences.

    Returns:
        List[Example]: A list of prepared examples.
    """
    return [
        Example(
            tokens=toks,
            activations=acts,
            normalized_activations=None
        )
        for toks, acts in zip(tokens, activations)
    ]

@dataclass
class Feature:
    """
    A feature extracted from a model's activations.

    Attributes:
        module_name (str): The module name associated with the feature.
        feature_index (int): The index of the feature within the module.
    """
    module_name: str
    feature_index: int

    def __repr__(self) -> str:
        """
        Return a string representation of the feature.

        Returns:
            str: A string representation of the feature.
        """
        return f"{self.module_name}_feature{self.feature_index}"


class FeatureRecord:
    """
    A record of feature data.

    Attributes:
        feature (Feature): The feature associated with the record.
    """

    def __init__(
        self,
        feature: Feature,
    ):
        """
        Initialize the feature record.

        Args:
            feature (Feature): The feature associated with the record.
        """
        self.feature = feature
        self.examples = []
        self.train = []
        self.test = []

    @property
    def max_activation(self):
        """
        Get the maximum activation value for the feature.

        Returns:
            float: The maximum activation value.
        """
        return self.examples[0].max_activation

    def save(self, directory: str, save_examples=False):
        """
        Save the feature record to a file.

        Args:
            directory (str): The directory to save the file in.
            save_examples (bool): Whether to save the examples. Defaults to False.
        """
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
        """
        Display the feature record in a formatted string.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to use for decoding.
            threshold (float): The threshold for highlighting activations. Defaults to 0.0.
            n (int): The number of examples to display. Defaults to 10.

        Returns:
            str: The formatted string.
        """
        from IPython.core.display import HTML, display

        def _to_string(tokens: TensorType["seq"], activations: TensorType["seq"]) -> str:
            """
            Convert tokens and activations to a string.

            Args:
                tokens (TensorType["seq"]): The tokenized input sequence.
                activations (TensorType["seq"]): The activation values.

            Returns:
                str: The formatted string.
            """
            result = []
            i = 0

            max_act = activations.max()
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
