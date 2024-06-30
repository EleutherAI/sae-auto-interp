import asyncio
from nnsight import LanguageModel 
from tqdm import tqdm

from sae_auto_interp.clients import get_client
from sae_auto_interp.scorers.scorer import ScorerInput, FuzzingScorer
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.utils import get_samples, load_tokenized_data, execute_model
from sae_auto_interp.features import feature_loader, Feature, FeatureRecord

# Load model and autoencoders
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
ae_dict, submodule_dict, edits = load_autoencoders(
    model, 
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2"
)

# Load tokenized data
tokens = load_tokenized_data(model.tokenizer)

# Load features I want to explain
samples = get_samples(features_per_layer=10)
samples = {layer : samples[layer] for layer in samples if int(layer) in [0]}
features = Feature.from_dict(samples)


# Raw features contains locations
raw_features_path = "/share/u/caden/sae-auto-interp/raw_features"
# Processed features contains extra information like logits, etc.
# This is split so we don't have to compute large matrix multiplications
# when VLLM is taking up most of the GPU
processed_features_path = "/share/u/caden/sae-auto-interp/processed_features"

scorer_inputs = []

for feature in features:
    record = FeatureRecord.load_record(feature, tokens, model.tokenizer, raw_features_path, processed_features_path)

    # Skip features with no activations
    if record.examples is None:
        continue
    scorer_inputs.append(
        ScorerInput(
            explanation="NONE",
            test_examples=record.examples[:5],
            record=record
        )
    )

client = get_client("local", "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit")
scorer = FuzzingScorer(client)

# Run the scorer. Execute model should automatically async 
# and batch a bunch of requests to the server.
scorer_out_dir = "/share/u/caden/sae-auto-interp/saved_scores"
asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs,
        output_dir=scorer_out_dir,
    )
)