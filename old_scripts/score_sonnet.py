# %%
import os
os.environ["CONFIG_PATH"] = "/mnt/ssd-1/gpaulo/SAE-Zoology/sae_auto_interp/configs/gpaulo_llama_256.yaml"

import asyncio
from nnsight import LanguageModel
from tqdm import tqdm
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import load_tokenized_data,execute_model
from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_top_and_quantiles
from sae_auto_interp.logger import logger
import torch
import argparse
import random

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
#layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]
layers = [int(args.layers)]

model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="cpu", dispatch=True,torch_dtype =torch.bfloat16)
n_tokens = 10_400_000
dataset_repo =  "kh4dien/fineweb-100m-sample"
batch_len = 256
tokens  = load_tokenized_data(model.tokenizer,n_tokens=n_tokens, dataset_repo=dataset_repo, batch_len=batch_len,dataset_split="train")

raw_features_path = "raw_features_llama"



def load_explanation(explanation_dir,feature):
    explanations_path = f"{explanation_dir}/layer{feature.layer_index}_feature{feature.feature_index}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    return explanation


scorer_inputs=[]
for layer in layers:
    records = FeatureRecord.from_tensor(
        tokens,
        layer_index=layer,
        selected_features=torch.arange(0,50),
        raw_dir= raw_features_path,
        processed_dir="processed_features_llama",
        max_examples=100000
    )
    
    for record in tqdm(records):

        try:
            
            explanation = load_explanation("saved_explanations/llama_sonnet",record.feature)
            record.tokenizer = model.tokenizer
            _, test = sample_top_and_quantiles(
                record=record,
                n_train=0,
                n_test=7,
                n_quantiles=10,
                seed=22,
                n_extra=0
            )
            non_activating = []
            while len(non_activating)<20:
                
                feature_idx = random.randint(0,len(records)-1)
                if feature_idx == record.feature.feature_index:
                    continue
                idx = random.randint(0,len(records[feature_idx].examples)-1)
                example = records[feature_idx].examples[idx]
                example.decode(model.tokenizer)
                non_activating.append(example)
            extra = []
            for quantiles in test:
                extra.extend(quantiles[-2:])
                quantiles = quantiles[:-2]
            
        except Exception as e:
            logger.error(f"Failed while sampling for {record.feature}: {e}") 
            continue

    
        scorer_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                extra_examples=extra,
                random_examples=non_activating,
                explanation=explanation
            )
        )

client = get_client("outlines", "casperhansen/llama-3-70b-instruct-awq", base_url="http://127.0.0.1:8000")
scorer = FuzzingScorer(client)
scorer_out_dir = "scores/llama_sonnet"
print("Running 1")
asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs,
        output_dir=scorer_out_dir,
        record_time=True
    )
)
