import asyncio
from nnsight import LanguageModel 

from sae_auto_interp.clients import get_client
from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.utils import get_samples, load_tokenized_data, execute_model
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_top_and_quantiles_single
from sae_auto_interp.logger import logger

# Load model and autoencoders
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
ae_dict, submodule_dict, edits = load_autoencoders(
    model, 
    list(range(12)),
    "saved_autoencoders/gpt2"
)

# Load tokenized data
tokens = load_tokenized_data(model.tokenizer)

# Load features to explain
samples = get_samples(features_per_layer=500)

# Raw features contains locations
raw_features_path = "raw_features"
# Processed features contains extra information like logits, etc.
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
        model.tokenizer,
        layer,
        selected_features=samples[layer],
        raw_dir= raw_features_path,
        processed_dir=processed_features_path,
        max_examples=2000
    )
    
    for record in records:
        if type(record) is not FeatureRecord:
            continue

        explanation = load_explanation(record.feature)

        try:
            _, test = sample_top_and_quantiles_single(record)
        except:
            logger.info("Not enough examples for record")
            continue

        scorer_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation
            )
        )

client = get_client("local", "meta-llama/Meta-Llama-3-8B-Instruct")
scorer = FuzzingScorer(client)
scorer_out_dir = "saved_scores"

# Run the scorer. Execute model should automatically async 
# and batch a bunch of requests to the server.
asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs,
        output_dir=scorer_out_dir,
    )
)