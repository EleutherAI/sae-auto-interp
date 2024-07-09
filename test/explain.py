import os
import asyncio
import random

from nnsight import LanguageModel
from keys import openrouter_key

os.environ["CONFIG_PATH"] = "configs/caden_gpt2.yaml"

from sae_auto_interp.explainers import SimpleExplainer, ExplainerInput
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import execute_model, load_tokenized_data
from sae_auto_interp.features import FeatureRecord

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
explainer_out_dir = "explanations/simple_local_70b"
explainer_inputs=[]
random.seed(22)

for layer in range(0,12,2):
    records = FeatureRecord.from_tensor(
        tokens,
        tokenizer=model.tokenizer,
        layer_index=layer,
        selected_features=list(range(50)),
        raw_dir= raw_features_path,
        min_examples=120,
        max_examples=10000
    )

    for record in records:

        examples = record.examples

        if len(examples) < 120:
            continue

        train_examples = random.sample(examples[:100], 10)
        
        explainer_inputs.append(
            ExplainerInput(
                train_examples=train_examples,
                record=record
            )
        )

# client = get_client("openrouter", "meta-llama/llama-3-70b-instruct", api_key=openrouter_key)
client = get_client("local", "meta-llama/Meta-Llama-3-8B-Instruct")

explainer = SimpleExplainer(client)

asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
    )
)

