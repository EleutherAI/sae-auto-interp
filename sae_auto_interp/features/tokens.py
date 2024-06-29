from transformer_lens import utils
from datasets import load_dataset
from transformers import AutoTokenizer

from ..import cache_config as CONFIG

def load_tokens(
    tokenizer: AutoTokenizer
):
    data = load_dataset(CONFIG.dataset_repo, split=CONFIG.dataset_split)

    tokens = utils.tokenize_and_concatenate(
        data, 
        tokenizer, 
        max_length=CONFIG.batch_len
    )   

    tokens = tokens.shuffle(CONFIG.seed)['tokens']