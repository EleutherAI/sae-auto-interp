# %%

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
from sae_auto_interp import cache_config as CONFIG
import random

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
#layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]
layers = [int(args.layers)]

model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="cpu", dispatch=True,torch_dtype =torch.bfloat16)
CONFIG.n_tokens = 10_400_000
CONFIG.dataset_repo =  "kh4dien/fineweb-100m-sample"
CONFIG.batch_len = 256

tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features_llama"



def load_explanation(explanation_dir,feature):
    explanations_path = f"{explanation_dir}/layer{feature.layer_index}_feature{feature.feature_index}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    return explanation


scorer_inputs_1=[]
scorer_inputs_2=[]
scorer_inputs_3=[]
scorer_random=[]
for layer in layers:
    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=model.tokenizer,
        selected_features=torch.arange(50,150),
        raw_dir= raw_features_path,
        n_random=30,
        min_examples=200,
        max_examples=10000
    )
    
    for record in tqdm(records):

        try:
            idx = random.sample(range(len(records)),1)[0]

            random_explanation = load_explanation("saved_explanations/llama_1",records[idx].feature)
            
            explanation_1 = load_explanation("saved_explanations/llama_1",record.feature)
            explanation_2 = load_explanation("saved_explanations/llama_2",record.feature)
            explanation_3 = load_explanation("saved_explanations/llama_3",record.feature)
            
            _, test, extra = sample_top_and_quantiles(
                record=record,
                n_train=0,
                n_test=10,
                n_quantiles=5,
                seed=22,
                n_extra=40
            )
        
        except Exception as e:
            logger.error(f"Failed while sampling for {record.feature}: {e}") 
            continue

        record.extra = extra

        scorer_inputs_1.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation_1
            )
        )
        scorer_inputs_2.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation_2
            )
        )
        scorer_inputs_3.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation_3
            )
        )
        scorer_random.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=random_explanation
            )
        )

client = get_client("local", "casperhansen/llama-3-70b-instruct-awq", base_url="http://127.0.0.1:8000")
scorer = FuzzingScorer(client)
scorer_out_dir = "scores/llama_1"
print("Running 1")
asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs_1,
        output_dir=scorer_out_dir,
    )
)
client = get_client("local", "casperhansen/llama-3-70b-instruct-awq", base_url="http://127.0.0.1:8001")
scorer = FuzzingScorer(client)

scorer_out_dir = "scores/llama_2"
print("Running 2")
asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs_2,
        output_dir=scorer_out_dir,
    )
)
client = get_client("local", "casperhansen/llama-3-70b-instruct-awq", base_url="http://127.0.0.1:8002")
scorer = FuzzingScorer(client)

scorer_out_dir = "scores/llama_3"
print("Running 3")
asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs_3,
        output_dir=scorer_out_dir,
    )
)
client = get_client("local", "casperhansen/llama-3-70b-instruct-awq", base_url="http://127.0.0.1:8000")
scorer = FuzzingScorer(client)

scorer_out_dir = "scores/llama_random"
print("Running random")
asyncio.run(
    execute_model(
        scorer, 
        scorer_random,
        output_dir=scorer_out_dir,
    )
)