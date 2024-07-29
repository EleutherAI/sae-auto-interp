import asyncio
from nnsight import LanguageModel
from tqdm import tqdm

from sae_auto_interp.clients import get_client,execute_model
from sae_auto_interp.scorers import ScorerInput, OpenAISimulator
from sae_auto_interp.utils import load_tokenized_data 
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_top_and_quantiles
from sae_auto_interp.logger import logger
import argparse
import json

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]

model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
tokens = load_tokenized_data(model.tokenizer,dataset_split="train")

raw_features_path = "raw_features"
processed_features_path = "processed_features"

def load_explanation(explanations_dir,feature):
    explanations_path = f"{explanations_dir}/layer{feature}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    return json.loads(explanation)
def flatten(l):
    return [item for sublist in l for item in sublist]



scorer_inputs = []

for layer in layers:
    module_name = f".transformer.h.{layer}"

    records = FeatureRecord.from_tensor(
        tokens,
        module_name,
        selected_features=list(range(0,50)),
        raw_dir = raw_features_path,
        processed_dir = processed_features_path,
        min_examples=200,
        max_examples=10000
    )
    
    for record in tqdm(records):

        try:
            split_top_200_20_all_20_explanation = load_explanation("saved_explanations/gpt2_split_top_200_20_all_20",record.feature)
            record.tokenizer = model.tokenizer
        
            _, test = sample_top_and_quantiles(
                record=record,
                n_train=0,
                n_test=4,
                n_quantiles=4,
                seed=22,
            )
            test = flatten(test)
            for example in test:
                example.decode(model.tokenizer)
        except Exception as e:
            logger.error(f"Failed while sampling for {record.feature}: {e}") 
            continue
        
        scorer_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=split_top_200_20_all_20_explanation
            )
        )

client = get_client("local", "casperhansen/llama-3-70b-instruct-awq", base_url="http://localhost:8001/v1")
scorer = OpenAISimulator(client)
scorer_out_dir = "scores/gpt2_sim"
asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs,
        output_dir=scorer_out_dir,
        record_time=True
    )
)