# %%
import asyncio
import random
import json
from nnsight import LanguageModel

from sae_auto_interp.scorers import NeighborScorer, ScorerInput
from sae_auto_interp.clients import get_client, execute_model
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.scorers.neighbor.utils import load_neighbors 

import argparse
import json

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]


model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
explanations_dir = "saved_explanations/gpt2_simple"
scorer_out_dir = "scores/neighbors"
random.seed(22)

with open("neighbors/unique.json", "r") as f:
    unique = json.load(f)

def load_explanation(feature):
    explanations_path = f"{explanations_dir}/{feature}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    if type(explanation) == dict:
        explanation = explanation["result"]

    return explanation

scorer_inputs = []

for layer in layers:
    module_name = f".transformer.h.{layer}"

    all_records = FeatureRecord.from_tensor(
        tokens,
        module_name,
        selected_features=list(range(0,50)),
        raw_dir = raw_features_path,
        min_examples=120,
        max_examples=10000
    )

    records = all_records[:10]
    load_neighbors(records, all_records, module_name, "neighbors/neighbors.json")

    for record in records:

        try:
            examples = record.examples
            test_examples = random.sample(examples[100:200], 20)
            explanation = load_explanation(record.feature)
            scorer_inputs.append(
                ScorerInput(
                    explanation=explanation,
                    record=record,
                    test_examples=test_examples,
                )
            )
        except Exception as e:
            print(e)
            continue

    break

client = get_client("local", "casperhansen/llama-3-70b-instruct-awq", base_url="http://localhost:8002/v1")

scorer = NeighborScorer(
    client,
    model.tokenizer
)

asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs,
        output_dir=scorer_out_dir,
        record_time=True
    )
)

