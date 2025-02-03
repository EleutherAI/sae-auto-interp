from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import cast
from torchtyping import TensorType
from typing import Any, Type, TypeVar, cast

def load_tokenized_data(
    ctx_len: int,
    tokenizer: AutoTokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_repo: str,
    dataset_split: str,
    dataset_name: str = "",
    dataset_row: str = "raw_content",
    seed: int = 22,
):
    """
    Load a huggingface dataset, tokenize it, and shuffle.
    """
    from datasets import load_dataset
    from transformer_lens import utils
    print(dataset_repo,dataset_name,dataset_split)
    data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)
    tokens_ds = utils.tokenize_and_concatenate(data, tokenizer, max_length=ctx_len,column_name=dataset_row)
    tokens_ds = tokens_ds.shuffle(seed)

    tokens = cast(TensorType["batch_size", "ctx_len"], tokens_ds["tokens"])

    return tokens


def load_filter(path: str, device: str = "cuda:0"):
    import json

    import torch

    with open(path) as f:
        filter = json.load(f)

    return {key: torch.tensor(value, device=device) for key, value in filter.items()}




def load_tokenizer(model):
    """
    Loads tokenizer to the default NNsight configuration.
    """

    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    tokenizer._pad_token = tokenizer._eos_token if hasattr(tokenizer, "_eos_token") else tokenizer.eos_token

    return tokenizer


T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)
