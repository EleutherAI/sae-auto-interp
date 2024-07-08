# %%


import os
os.environ["CONFIG_PATH"] = "configs/caden_gpt2.yaml"

from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features.frequency import FrequencyCache
from nnsight import LanguageModel
import torch

layer = 0

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
print("Model loaded")

ae_dict, submodule_dict = load_autoencoders(
    model, 
    [layer],
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2",
)
print("Autoencoders loaded")


#  %%

cache = FrequencyCache(model, submodule_dict)
cache.run()
results = cache.save()


# %%

from transformer_lens import utils
from datasets import load_dataset

def load_tokenized_data(
    tokenizer
):
    data = load_dataset(
        "stas/openwebtext-10k", 
    )

    data = data['train']

    tokens = utils.tokenize_and_concatenate(
        data, 
        tokenizer, 
        max_length=1024
    )   

    tokens = tokens.shuffle(22)['tokens']

    return tokens

tokens = load_tokenized_data(model.tokenizer)
n_batches = 20
tokens = tokens[:n_batches]
# %%


data = {}

for layer, selected_features in results.items():
    with model.trace(tokens):

        ae_out = model.transformer.h[0].ae.output
        ae_out = torch.mean(ae_out, dim=0)
        ae_out = ae_out[:,selected_features]
        ae_out.save()

    data[layer] = ae_out.value.detach().cpu().numpy()

# %%

import matplotlib.pyplot as plt
plt.plot(data[0])