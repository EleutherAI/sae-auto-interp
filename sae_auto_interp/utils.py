from transformer_lens import utils
from datasets import load_dataset
from transformers import AutoTokenizer
from .features.constructors import pool_max_activation_windows, random_activation_windows
from torchtyping import TensorType
from .features import FeatureRecord

def load_tokenized_data(
    tokenizer: AutoTokenizer,
    dataset_repo: str = "kh4dien/fineweb-100m-sample",
    dataset_name: str = "",
    dataset_split: str = "train[:15%]",
    seq_len: int = 64,
    seed: int = 22,
):
    data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)

    tokens = utils.tokenize_and_concatenate(
        data, 
        tokenizer, 
        max_length=seq_len
    )   

    tokens = tokens.shuffle(seed)['tokens']

    return tokens


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
    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    tokenizer._pad_token = tokenizer._eos_token

    return tokenizer

def default_constructor(record, tokens, locations, activations):

        pool_max_activation_windows(
            record,
            tokens=tokens,
            locations=locations,
            activations=activations,
            k=200
        )

        random_activation_windows(
            record,
            tokens=tokens,
            locations=locations,
        )
