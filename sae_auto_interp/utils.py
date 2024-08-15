from transformers import AutoTokenizer


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




def load_tokenizer(model):
    """
    Loads tokenizer to the default NNsight configuration.
    """

    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    tokenizer._pad_token = tokenizer._eos_token

    return tokenizer
