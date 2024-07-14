# %%
import os
import asyncio
import random

from nnsight import LanguageModel
from keys import openrouter_key

os.environ["CONFIG_PATH"] = "configs/gpt2_128k.yaml"

import json
from sae_auto_interp.explainers import SimpleExplainer, ExplainerInput
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import execute_model, load_tokenized_data
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.features.sampling import sample_top_and_quantiles

from sae_auto_interp.scorers.neighbor.utils import load_neighbors

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
explainer_out_dir = "results/explanations/test"
explainer_inputs=[]
random.seed(22)

with open("neighbors/unique.json", "r") as f:
    unique = json.load(f)

modules = [f".transformer.h.{i}" for i in range(0,12,2)]

for module_path in modules:
    layer_features = unique[module_path] 

    all_records = FeatureRecord.from_tensor(
        tokens,
        module_name=module_path,
        selected_features=layer_features,
        raw_dir = raw_features_path,
        min_examples=120,
        max_examples=10000
    )

    break

# %%

records = all_records[:10]

records = load_neighbors(records, all_records, module_path, "neighbors/neighbors.json")

# %%
import os
os.environ["CONFIG_PATH"] = "configs/gpt2_128k.yaml"
from sae_auto_interp.autoencoders import load_autoencoders

from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
submodule_dict = load_autoencoders(
    model, 
    [0],
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/OpenAI/gpt2_128k",
)


# %%

neighbors = list(records[0].neighbors.values())

# %%

neighbors[1].display(model.tokenizer)

print(neighbors[1].feature)

# %%

prompt = 'the attention of the media and the players? Nobody ever heard about it." An ATP spokesman said'
with model.trace(prompt):

    val = model.transformer.h[0].ae.output.save()

print(val.value[:,:,118117])
print(val.value[:,:,84355])

# %%

print(neighbors[4].feature)
print(neighbors[5].feature)

    # records = load_neighbors(records, records, "neighbors.json")

    # for record in records:

    #     examples = record.examples

    #     print(len(examples))
    #     continue
    #     train_examples = random.sample(examples[:100], 10)
        
    #     explainer_inputs.append(
    #         ExplainerInput(
    #             train_examples=train_examples,
    #             record=record
    #         )
    #     )



# client = get_client("local", "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit")

# explainer = SimpleExplainer(client, tokenizer=model.tokenizer)

# asyncio.run(
#     execute_model(
#         explainer, 
#         explainer_inputs,
#         output_dir=explainer_out_dir,
#         record_time=True
#     )
# )

