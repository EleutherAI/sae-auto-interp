import datasets
import json
from typing import List
from transformer_lens import utils

def get_batches(config: dict[str,str],tokenizer):
    print("Loading dataset")
    ## This is hardcoded for now but I should make a dataset loader
    dataset = datasets.load_dataset(
        config["dataset_repo"], name=config["dataset_name"], split=config["split"]
    )

    keep_examples = config["number_examples"]
    dataset = dataset.select(range(keep_examples))
    tokenized_data = utils.tokenize_and_concatenate(
        dataset, tokenizer, max_length=256
    )
    all_tokens = tokenized_data["tokens"]

    batch_size = config["batch_size"]
    mini_batches = all_tokens.split(batch_size)
    mini_batches = [batch for batch in mini_batches]
    mini_batches = mini_batches[:-1]

    number_of_tokens= mini_batches.shape[0] * mini_batches.shape[1]
    print("Collecting features over {} tokens".format(number_of_tokens))

    return mini_batches

def get_available_configs()->List[str]:
    with open("config.json") as f:
        config = json.load(f)
    config_names = list(config.keys())
    return config_names

def get_config(config_name:str)->dict[str,str]:
    with open("config.json") as f:
        config = json.load(f)
    return config[config_name]


