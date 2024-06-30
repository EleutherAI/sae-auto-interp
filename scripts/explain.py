import asyncio
from nnsight import LanguageModel 
from tqdm import tqdm

from sae_auto_interp.explainers import ChainOfThought, ExplainerInput
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import execute_model, load_tokenized_data, get_samples
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import CombinedStat, Logits, feature_loader, Feature, FeatureRecord

# Load model and autoencoders
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
ae_dict, submodule_dict, edits = load_autoencoders(
    model, 
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2"
)

# Load tokenized data
tokens = load_tokenized_data(model.tokenizer)

# Load features to explain
samples = get_samples(features_per_layer=20)
samples = {layer : samples[layer] for layer in samples if int(layer) in [0,2,4,6,8,10]}
features = Feature.from_dict(samples)

# Raw features contains locations
raw_features_path = "/share/u/caden/sae-auto-interp/raw_features"
# Processed features contains extra information like logits, etc.
# This is split so we don't have to compute large matrix multiplications
# when VLLM is taking up most of the GPU
processed_features_path = "/share/u/caden/sae-auto-interp/processed_features"

explainer_inputs = []

for feature in tqdm(features):
    record = FeatureRecord.load_record(
        feature, 
        tokens, 
        model.tokenizer, 
        feature_dir=raw_features_path, 
        processed_dir=processed_features_path
    )

    # Skip features with no activations
    if record.examples is None:
        continue
    explainer_inputs.append(
        ExplainerInput(
            train_examples=record.examples[:10],
            record=record
        )
    )

client = get_client("local", "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit")
explainer = ChainOfThought(client)
explainer_out_dir = "/share/u/caden/sae-auto-interp/saved_explanations/caden"

# Run the explainer. Execute model should automatically async 
# and batch a bunch of requests to the server.
asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
    )
)