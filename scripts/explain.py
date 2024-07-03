#%%

import asyncio
from nnsight import LanguageModel 
from tqdm import tqdm

from sae_auto_interp.explainers import ChainOfThought, ExplainerInput
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import execute_model, load_tokenized_data, get_samples
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments.sampling import sample_top_and_quantiles_single
import random


# Load model and autoencoders
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
# ae_dict, submodule_dict, edits = load_autoencoders(
#     model, 
#     list(range(0,12,2)),
#     "sae_auto_interp/autoencoders/oai/gpt2" 
# )

tokens = load_tokenized_data(model.tokenizer)
samples = get_samples(features_per_layer=200)

raw_features_path = "raw_features"
processed_features_path = "processed_features"

explainer_inputs = []
random.seed(22)


for layer in range(6,12,2):
    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=model.tokenizer,
        selected_features=samples[layer],
        raw_dir=raw_features_path,
        processed_dir=processed_features_path,
        min_examples=300,
        max_examples=2000
    )

    n = 0
    for record in records:
        if type(record) is str:
            continue

        train_set = random.sample(record.examples[:50], 10)

        explainer_inputs.append(
            ExplainerInput(
                train_examples=record.examples[:50],
                record=record
            )
        )
        n += 1
        if n > 150:
            break

client = get_client("local", "casperhansen/llama-3-70b-instruct-awq")
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