from typing import Dict, List

from torchtyping import TensorType
from transformers import AutoTokenizer

from .features import FeatureRecord
from .features.constructors import pool_max_activation_windows, random_activation_windows

def load_tokenized_data(
    tokenizer: AutoTokenizer,
    dataset_repo: str = "kh4dien/fineweb-100m-sample",
    dataset_name: str = "",
    dataset_split: str = "train[:15%]",
    seq_len: int = 64,
    seed: int = 22,
):
    """
    Load a huggingface dataset, tokenize it, and shuffle.
    """
    from transformer_lens import utils
    from datasets import load_dataset

    data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)

    tokens = utils.tokenize_and_concatenate(
        data, 
        tokenizer, 
        max_length=seq_len
    )   

    tokens = tokens.shuffle(seed)['tokens']

    return tokens


def load_filter(filter: Dict[str, List[int]], device="cuda:0") -> Dict:
    """
    Wrap a filter dictionary in torch tensors.
    """
    import torch

    return {
        key : torch.tensor(value, device=device) 
        for key, value in filter.items()
    }


def display(
    record: FeatureRecord,
    tokenizer: AutoTokenizer,
    threshold: float = 0.0,
    n: int = 10
) -> str:

    from IPython.core.display import display, HTML

    def _to_string(
        tokens: TensorType["seq"], 
        activations: TensorType["seq"]
    ) -> str:
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
        _to_string(
            tokenizer.batch_decode(example.tokens), 
            example.activations
        )
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


def default_constructor(
    record: FeatureRecord, 
    tokens: TensorType["batch", "seq"], 
    locations: TensorType["locations", 2],
    activations: TensorType["locations"],
    n_random: int,
    ctx_len: int,
    max_examples: int
):

    pool_max_activation_windows(
        record,
        tokens=tokens,
        locations=locations,
        activations=activations,
        ctx_len=ctx_len,
        max_examples=max_examples,
    )

    random_activation_windows(
        record,
        tokens=tokens,
        locations=locations,
        n_random=n_random,
        ctx_len=ctx_len,
    )
