# %%
import os
import asyncio
import random

from nnsight import LanguageModel
from keys import openrouter_key

os.environ["CONFIG_PATH"] = "configs/pythia.yaml"

from sae_auto_interp.explainers import SimpleExplainer, ExplainerInput
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import execute_model, load_tokenized_data
from sae_auto_interp.features import FeatureRecord

model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
explainer_out_dir = "explanations/claude"
explainer_inputs=[]
random.seed(22)

# %%

records = FeatureRecord.from_tensor(
        tokens,
        tokenizer=model.tokenizer,
        module_name='.gpt_neox.layers.4.attention',
        raw_dir= raw_features_path,
        min_examples=10,
        max_examples=10000
    )
# %%

FeatureRecord.display(records[0].examples[:20], threshold=0.4)
# records[0].examples[0].activations

# %%

with model.trace("test"):

    val = model.gpt_neox.layers[0].output.save()

val.shape

# %%
# max_acts = []

# import matplotlib.pyplot as plt

# plt.hist(records, bins=100)

# %%

records[0].examples = records[0].decode(records[0].examples, tokenizer=model.tokenizer)
FeatureRecord.display(records[0].examples[:20])

# %%
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
    break

client = get_client("openrouter", "anthropic/claude-3-haiku", api_key=openrouter_key)
# client = get_client("local", "casperhansen/llama-3-70b-instruct-awq")

explainer = SimpleExplainer(client)

asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
        record_time=True
    )
)

