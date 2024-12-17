from transformers import AutoTokenizer, default_data_collator
from typing import List, Dict
import torch

def packed_data_collator(input_ids: List[Dict], return_tensors: str = "pt"):
    """
    Collates input IDs into a batch.

    Args:
        input_ids (List[Dict]): The input IDs to collate.
        return_tensors (str, optional): The type of tensors to return. Defaults to "pt".

    Returns:
        Dict: The collated batch.
    """
    batch = {"input_ids": [], "position_ids": []}
    for input_id in input_ids:
        batch["input_ids"] += input_id
        batch["position_ids"] += list(range(len(input_id)))
    
    return default_data_collator([batch], return_tensors=return_tensors)

def load_dataset(
    ctx_len: int,
    tokenizer: AutoTokenizer,
    dataset_repo: str,
    dataset_split: str,
    dataset_name: str = "",
    column_name: str = "raw_content",
    seed: int = 22,
) -> torch.Tensor:
    """
    Load a Hugging Face dataset, tokenize it, shuffle it, and pack it.

    Args:
        ctx_len (int): Context length for tokenization.
        tokenizer (AutoTokenizer): The tokenizer to use.
        dataset_repo (str): The dataset repository name.
        dataset_split (str): The dataset split to use.
        dataset_name (str, optional): The dataset name. Defaults to "".
        column_name (str, optional): The column name to use for tokenization. Defaults to "text".
        seed (int, optional): Random seed for shuffling. Defaults to 22.

    Returns:
        torch.Tensor: The tokenized, shuffled, and packed dataset.
    """
    from datasets import load_dataset
    dataset = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)
    dataset = dataset.shuffle(seed)
    dataset = dataset.map(lambda x: tokenizer(x[column_name], add_special_tokens=False, return_attention_mask=False), batched=True)
    dataset = dataset.map(lambda x: packed_data_collator(x['input_ids']), batched=True, batch_size=10, remove_columns=dataset.column_names) # TODO: Change hardcoded batch size

    return dataset

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
