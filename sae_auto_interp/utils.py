from torchtyping import TensorType
from transformers import AutoTokenizer

from .features import FeatureRecord


def load_tokenized_data(
    ctx_len: int,
    tokenizer: AutoTokenizer,
    dataset_repo: str,
    dataset_split: str,
    dataset_name: str = "",
    seed: int = 22,
):
    """
    Load a huggingface dataset, tokenize it, and shuffle.
    """
    from datasets import load_dataset
    from transformer_lens import utils

    data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)

    tokens = utils.tokenize_and_concatenate(data, tokenizer, max_length=ctx_len)

    tokens = tokens.shuffle(seed)["tokens"]

    return tokens


def load_filter(path: str, device: str = "cuda:0"):
    import json

    import torch

    with open(path) as f:
        filter = json.load(f)

    return {key: torch.tensor(value, device=device) for key, value in filter.items()}


def display(
    record: FeatureRecord, tokenizer: AutoTokenizer, threshold: float = 0.0, n: int = 10
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
        for example in record.examples[:n]
    ]

    display(HTML("<br><br>".join(strings)))


def load_tokenizer(model):
    """
    Loads tokenizer to the default NNsight configuration.
    """

    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    tokenizer._pad_token = tokenizer._eos_token

    return tokenizer
