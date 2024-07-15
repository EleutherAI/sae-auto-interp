# %%
import os
import asyncio
import random

from nnsight import LanguageModel
from keys import openrouter_key

import json
from sae_auto_interp.explainers import SimpleExplainer, ExplainerInput
from sae_auto_interp.clients import get_client, execute_model
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features import FeatureRecord

from sae_auto_interp.scorers.neighbor.utils import load_neighbors

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
explainer_out_dir = "results/explanations/simple"
explainer_inputs=[]
random.seed(22)


for layer in range(0,12,2):
    module_name = f".transformer.h.{layer}"

    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        module_name,
        selected_features=[0,1,3,4],
        raw_dir = raw_features_path,
        min_examples=120,
        max_examples=10000
    )

    for record in records:

        examples = record.examples
        train_examples = random.sample(examples[:100], 10)

        record.top_logits = None
        
        explainer_inputs.append(
            ExplainerInput(
                train_examples=train_examples,
                record=record
            )
        )

client = get_client("local", "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit")

explainer = SimpleExplainer(client, tokenizer=model.tokenizer)

asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
        record_time=True
    )
)

