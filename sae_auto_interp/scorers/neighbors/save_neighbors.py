# %%

from nnsight import LanguageModel 
import os 

os.environ["CONFIG_PATH"] = "configs/gpt2_128k.yaml"

from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import CombinedStat, FeatureRecord, Logits, Activation, QuantileSizes

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
submodule_dict = load_autoencoders(
    model, 
    list(range(0,12,2)),
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/OpenAI/gpt2_128k",
)

# %%
import torch
import torch.nn.functional as F

tensor = submodule_dict[0].ae.autoencoder._module.decoder.weight

def cos(matrix, n=1000):
    
    a = matrix[:,:n]
    b = matrix   

    a = F.normalize(a, p=2, dim=0)
    b = F.normalize(b, p=2, dim=1)

    cos_sim = torch.mm(a.t(), b)

    return cos_sim

import json

data = {}

for i in range(0,12,2):
    tensor = submodule_dict[i].ae.autoencoder._module.decoder.weight
    cos_sim = cos(tensor)
    top = torch.topk(cos_sim[:100], k=10)
    unique_top = torch.unique(top.indices)
    print(len(torch.unique(unique_top)))


    # break
    data[submodule_dict[i]._module_path] = unique_top.tolist()
    
# %%
list(data.values())[0]

# %%
with open("neighbors.json", "w") as f:
    json.dump(data, f)
