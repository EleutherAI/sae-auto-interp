from transformer_lens import utils
from datasets import load_dataset
from transformers import AutoTokenizer

def load_tokenized_data(
    tokenizer: AutoTokenizer,
    dataset_repo: str = "kh4dien/fineweb-100m-sample",
    dataset_name: str = "",
    dataset_split: str = "train[:15%]",
    seq_len: int = 64,
    seed: int = 22,
):
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

