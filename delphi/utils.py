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
    add_bos_token: bool = True,
):
    """
    Load a huggingface dataset, tokenize it, and shuffle.
    """
    from datasets import load_dataset
    from sparsify.data import chunk_and_tokenize
    
    print(dataset_repo,dataset_name,dataset_split)

    data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)
    tokens_ds = chunk_and_tokenize(data, tokenizer, max_seq_len=ctx_len, text_key=dataset_row)
    tokens_ds = tokens_ds.shuffle(seed)

    tokens = cast(TensorType["batch", "seq"], tokens_ds["input_ids"])
    
    return tokens


def load_filter(path: str, device: str = "cuda:0"):
    import json

    import torch

    with open(path) as f:
        filter = json.load(f)

    return {key: torch.tensor(value, device=device) for key, value in filter.items()}



T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)
