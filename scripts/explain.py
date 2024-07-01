import asyncio
from nnsight import LanguageModel 
from tqdm import tqdm

from sae_auto_interp.explainers import ChainOfThought, ExplainerInput
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import execute_model, load_tokenized_data, get_samples
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import FeatureRecord

# Load model and autoencoders
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
ae_dict, submodule_dict, edits = load_autoencoders(
    model, 
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

explainer_inputs = []


for layer in [0,1,2,3,4,5,6,7,8,9,10,11]:
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
        explainer_inputs.append(
            ExplainerInput(
                train_examples=record.examples[:10],
                record=record
            )
        )

client = get_client("local", "meta-llama/Meta-Llama-3-8B-Instruct")
explainer = ChainOfThought(client)
explainer_out_dir = "saved_explanations/cot"

# Run the explainer. Execute model should automatically async 
# and batch a bunch of requests to the server.
asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
    )
)