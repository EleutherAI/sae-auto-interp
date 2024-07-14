from typing import List, Callable, Awaitable
# from .scorers.scorer import ScorerInput
# from .explainers import ExplainerInput

from transformer_lens import utils
from datasets import load_dataset
from transformers import AutoTokenizer

from . import cache_config as CONFIG



def load_tokenized_data(
    tokenizer: AutoTokenizer,
    config=CONFIG,
    **kwargs
):
    # Use kwargs to override config values if provided
    dataset_repo = kwargs.get('dataset_repo', config.dataset_repo)
    dataset_name = kwargs.get('dataset_name', config.dataset_name)
    dataset_split = kwargs.get('dataset_split', config.dataset_split)
    seq_len = kwargs.get('seq_len', config.seq_len)
    seed = kwargs.get('seed', config.seed)

    # Load the dataset
    data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)

    # Tokenize and concatenate
    tokens = utils.tokenize_and_concatenate(
        data, 
        tokenizer, 
        max_length=seq_len
    )   

    # Shuffle the tokens
    tokens = tokens.shuffle(seed)['tokens']

    return tokens

