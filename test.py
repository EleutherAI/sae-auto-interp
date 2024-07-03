# %%

import asyncio

from sae_auto_interp.clients import get_client
from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.utils import get_samples, load_tokenized_data, execute_model
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_quantiles
from sae_auto_interp.logger import logger
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

tokens = load_tokenized_data(tokenizer)
samples = get_samples(features_per_layer=10)

raw_features_path = "raw_features"
processed_features_path = "processed_features"
explanations_dir = "saved_explanations/cot"


def load_explanation(feature):
    explanations_path = f"{explanations_dir}/layer{feature.layer_index}_feature{feature.feature_index}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    return explanation

scorer_inputs = []

for layer in [0]:
    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=tokenizer,
        selected_features=samples[layer],
        raw_dir= raw_features_path,
        processed_dir=processed_features_path,
        n_random=10,
        min_examples=300,
        max_examples=2000
    )

# %%

records[0].display()

# %%

tokenizer.batch_decode(records[0].non_activating)

# %%

client = get_client("local", "casperhansen/llama-3-70b-instruct-awq")
scorer = FuzzingScorer(client)
scorer_out_dir = "saved_scores/new"

asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs,
        output_dir=scorer_out_dir,
    )
)