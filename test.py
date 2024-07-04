# %%

import asyncio
from transformers import AutoTokenizer

from sae_auto_interp.clients import get_client
from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.utils import load_tokenized_data, execute_model
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_top_and_quantiles
from sae_auto_interp.autoencoders.ae import load_autoencoders
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)    
tokens = load_tokenized_data(model.tokenizer)
ae_dict, submodule_dict = load_autoencoders(
    model, 
    list(range(0,12,2)),
    "sae_auto_interp/autoencoders/oai/gpt2"
)

raw_features_path = "raw_features"
processed_features_path = "processed_features"
explanations_dir = "saved_explanations/cot"
scorer_out_dir = "saved_scores/cot"



layer_records = {}

for layer in [0]:
    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=model.tokenizer,
        selected_features=list(range(10)),
        raw_dir= raw_features_path,
        processed_dir=processed_features_path,
        n_random=10,
        min_examples=200,
        max_examples=2000
    )
    
    for record in records:

        try:
            train, test, extra = sample_top_and_quantiles(
                record=record,
                n_train=20,
                n_test=5,
                n_quantiles=4,
                n_extra=10,
                seed=22,
            )
            print(extra)
        except:
            continue    


        record.train = train
        record.extra = extra


    layer_records[layer] = records

# %%

import torch

a = torch.tensor(0)

a.dim()