# %%

import asyncio
from nnsight import LanguageModel
from tqdm import tqdm
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import load_tokenized_data, get_samples,execute_model
from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_top_and_quantiles
from sae_auto_interp.logger import logger
import torch
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
#layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]
layers = [int(args.layers)]

model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="cpu", dispatch=True,torch_dtype =torch.bfloat16)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features_llama"

samples = get_samples(N_LAYERS=32,N_FEATURES=131072,N_SAMPLES=1000)

def load_explanation(explanation_dir,feature):
    explanations_path = f"{explanation_dir}/layer{feature.layer_index}_feature{feature.feature_index}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    return explanation

scorer_inputs_short = []
scorer_inputs_long = []
for layer in layers:
    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=model.tokenizer,
        selected_features=samples[layer],
        raw_dir= raw_features_path,
        n_random=10,
        min_examples=200,
        max_examples=2000
    )
    
    for record in tqdm(records):

        try:
            explanation_short = load_explanation("saved_explanations/llama_simple",record.feature)
            explanation_long = load_explanation("saved_explanations/llama_simple_long",record.feature)
            _, test, extra = sample_top_and_quantiles(
                record=record,
                n_train=20,
                n_test=5,
                n_quantiles=4,
                seed=22,
                n_extra=10
            )
            
        except Exception as e:
            logger.error(f"Failed while sampling for {record.feature}: {e}") 
            continue

        record.extra = extra

        scorer_inputs_short.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation_short
            )
        )
        scorer_inputs_long.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation_long
            )
        )

client = get_client("local", "casperhansen/llama-3-70b-instruct-awq", base_url="http://127.0.0.1:8001")
scorer = FuzzingScorer(client)
scorer_out_dir = "scores/llama_short"
print("Running short")
asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs_short,
        output_dir=scorer_out_dir,
    )
)
scorer_out_dir = "scores/llama_long"
print("Running long")
asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs_long,
        output_dir=scorer_out_dir,
    )
)