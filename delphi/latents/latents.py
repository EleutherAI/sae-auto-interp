from dataclasses import dataclass, field
from typing import Optional

import blobfile as bf
import orjson
from jaxtyping import Float
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class Example:
    """
    A single example of latent data.
    """

    tokens: Float[Tensor, "ctx_len"]
    """Tokenized input sequence."""

    activations: Float[Tensor, "ctx_len"]
    """Activation values for the input sequence."""

    normalized_activations: Optional[Float[Tensor, "ctx_len"]] = None
    """Activations quantized to integers in [0, 10]."""

    @property
    def max_activation(self) -> float:
        """
        Get the maximum activation value.

        Returns:
            float: The maximum activation value.
        """
        return float(self.activations.max())


def prepare_examples(
    tokens: Float[Tensor, "examples ctx_len"],
    activations: Float[Tensor, "examples ctx_len"],
) -> list[Example]:
    """
    Prepare a list of examples from input tokens and activations.

    Args:
        tokens: Tokenized input sequences.
        activations: Activation values for the input sequences.

    Returns:
        list[Example]: A list of prepared examples.
    """
    return [
        Example(tokens=toks, activations=acts, normalized_activations=None)
        for toks, acts in zip(tokens, activations)
    ]


@dataclass
class Latent:
    """
    A latent extracted from a model's activations.
    """

    module_name: str
    """The module name associated with the latent."""

    latent_index: int
    """The index of the latent within the module."""

    def __repr__(self) -> str:
        """
        Return a string representation of the latent.

        Returns:
            str: A string representation of the latent.
        """
        return f"{self.module_name}_latent{self.latent_index}"


@dataclass
class LatentRecord:
    """
    A record of latent data.
    """

    latent: Latent
    """The latent associated with the record."""

    examples: list[Example] = field(default_factory=list)
    """Example sequences where the latent activations, assumed to be sorted in
    descending order by max activation."""

    not_active: list[Example] = field(default_factory=list)
    """Non-activating examples."""

    train: list[Example] = field(default_factory=list)
    """Training examples."""

    test: list[list[Example]] = field(default_factory=list)
    """Test examples."""

    @property
    def max_activation(self) -> float:
        """
        Get the maximum activation value for the latent.

        Returns:
            float: The maximum activation value.
        """
        return self.examples[0].max_activation

    def save(self, directory: str, save_examples: bool = False):
        """
        Save the latent record to a file.

        Args:
            directory: The directory to save the file in.
            save_examples: Whether to save the examples. Defaults to False.
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
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        threshold: float = 0.0,
        n: int = 10,
    ):
        """
        Display the latent record in a formatted string.

        Args:
            tokenizer: The tokenizer to use for decoding.
            threshold: The threshold for highlighting activations.
                Defaults to 0.0.
            n: The number of examples to display. Defaults to 10.

        Returns:
            str: The formatted string.
        """
        from IPython.core.display import HTML, display

        def _to_string(tokens: list[str], activations: Float[Tensor, "ctx_len"]) -> str:
            """
            Convert tokens and activations to a string.

            Args:
                tokens: The tokenized input sequence.
                activations: The activation values.

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
            return ""

        strings = [
            _to_string(tokenizer.batch_decode(example.tokens), example.activations)
            for example in self.examples[:n]
        ]

        display(HTML("<br><br>".join(strings)))
