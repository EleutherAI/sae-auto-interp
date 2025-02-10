from dataclasses import dataclass
from typing import Optional
import blobfile as bf
import orjson
from torchtyping import TensorType
from transformers import AutoTokenizer


@dataclass
class Example:
    """
    A single example of latent data.

    Attributes:
        tokens (TensorType["seq"]): Tokenized input sequence.
        activations (TensorType["seq"]): Activation values for the input sequence.
        normalized_activations (TensorType["seq"]): Activations quantized to integers in [0, 10].
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
class Latent:
    """
    A latent extracted from a model's activations.

    Attributes:
        module_name (str): The module name associated with the latent.
        latent_index (int): The index of the latent within the module.
    """
    module_name: str
    latent_index: int

    def __repr__(self) -> str:
        """
        Return a string representation of the latent.

        Returns:
            str: A string representation of the latent.
        """
        return f"{self.module_name}_latent{self.latent_index}"


class LatentRecord:
    """
    A record of latent data.

    Attributes:
        latent (Latent): The latent associated with the record.
        examples: list[Example]: Example sequences where the latent activations,
          assumed to be sorted in descending order by max activation
    """

    def __init__(
        self,
        latent: Latent,
    ):
        """
        Initialize the latent record.

        Args:
            latent (Latent): The latent associated with the record.
        """
        self.latent = latent
        self.examples: list[Example] = []
        self.not_active: list[Example] = []
        self.train: list[list[Example]] = []
        self.test: list[list[Example]] = []

    @property
    def max_activation(self):
        """
        Get the maximum activation value for the latent.

        Returns:
            float: The maximum activation value.
        """
        return self.examples[0].max_activation

    def save(self, directory: str, save_examples=False):
        """
        Save the latent record to a file.

        Args:
            directory (str): The directory to save the file in.
            save_examples (bool): Whether to save the examples. Defaults to False.
        """
        path = f"{directory}/{self.latent}.json"
        serializable = self.__dict__

        if not save_examples:
            serializable.pop("examples")
            serializable.pop("train")
            serializable.pop("test")

        serializable.pop("latent")
        with bf.BlobFile(path, "wb") as f:
            f.write(orjson.dumps(serializable))
    

    def display(
        self,
        tokenizer: AutoTokenizer,
        threshold: float = 0.0,
        n: int = 10,
    ) -> str:
        """
        Display the latent record in a formatted string.

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
