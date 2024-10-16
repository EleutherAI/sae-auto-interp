from transformers import AutoTokenizer
import torch

def load_tokenized_data(
    ctx_len: int,
    tokenizer: AutoTokenizer,
    dataset_repo: str,
    dataset_split: str,
    dataset_name: str = "",
    column_name: str = "raw_content",
    seed: int = 22,
) -> torch.Tensor:
    """
    Load a Hugging Face dataset, tokenize it, and shuffle.

    Args:
        ctx_len (int): Context length for tokenization.
        tokenizer (AutoTokenizer): The tokenizer to use.
        dataset_repo (str): The dataset repository name.
        dataset_split (str): The dataset split to use.
        dataset_name (str, optional): The dataset name. Defaults to "".
        column_name (str, optional): The column name to use for tokenization. Defaults to "text".
        seed (int, optional): Random seed for shuffling. Defaults to 22.

    Returns:
        torch.Tensor: The tokenized and shuffled dataset.
    """
    from datasets import load_dataset
    from transformer_lens import utils
    data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)
    tokens = utils.tokenize_and_concatenate(data, tokenizer, max_length=ctx_len,column_name=column_name)

    tokens = tokens.shuffle(seed)["tokens"]

    return tokens


def load_filter(path: str, device: str = "cuda:0") -> dict:
    """
    Load a filter from a JSON file and convert values to tensors.

    Args:
        path (str): Path to the JSON file containing the filter.
        device (str, optional): The device to load the tensors to. Defaults to "cuda:0".

    Returns:
        dict: A dictionary with tensor values on the specified device.
    """
    import json

    import torch

    with open(path) as f:
        filter = json.load(f)

    return {key: torch.tensor(value, device=device) for key, value in filter.items()}




def load_tokenizer(model: str) -> AutoTokenizer:
    """
    Loads tokenizer to the default NNsight configuration.

    Args:
        model (str): The model name or path to load the tokenizer from.

    Returns:
        AutoTokenizer: The configured tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    tokenizer._pad_token = tokenizer._eos_token

    return tokenizer
