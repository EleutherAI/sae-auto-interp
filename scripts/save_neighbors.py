# %%
from nnsight import LanguageModel 
from sae_auto_interp.autoencoders.ae import load_autoencoders


model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
submodule_dict = load_autoencoders(
    model, 
    list(range(0,12,2)),
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/OpenAI/gpt2_128k",
)

# %%
import torch
import torch.nn.functional as F
from collections import defaultdict


def cos(matrix, n=1000):
    
    a = matrix[:,:n]
    b = matrix   

    a = F.normalize(a, p=2, dim=0)
    b = F.normalize(b, p=2, dim=0)

    cos_sim = torch.mm(a.t(), b)

    return cos_sim

import json

data = defaultdict(dict)
unique = {}


# n determines how many features we want to compare
# we use torch.unique to get all unique features that we need to cache 
# these are the neighbors
n = 100

for module, ae in submodule_dict.items():
    tensor = ae.ae.autoencoder._module.decoder.weight
    cos_sim = cos(tensor)
    top = torch.topk(cos_sim[:n], k=10)

    top_indices = top.indices
    top_values = top.values

    for i, (indices, values) in enumerate(zip(top_indices, top_values)):
        data[module][i] = {
            "indices": indices.tolist()[1:],
            "values": values.tolist()[1:]
        }
    
    unique[module] = torch.unique(top_indices).tolist()


# %%

# This creates two jsons

# Neighbors contains information about the neighbors and distances for n neighbors
with open("neighbors.json", "w") as f:
    json.dump(data, f)


# Unique contains all the features you'll need to cache to run neighbors. 
with open("unique.json", "w") as f:
    json.dump(unique, f)