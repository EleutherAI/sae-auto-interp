# %%
from nnsight import LanguageModel
import torch
from sae_auto_interp.autoencoders import load_autoencoders
from sae_auto_interp.features import FeatureCache
import json

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
submodule_dict = load_autoencoders(
    model, 
    # list(range(0,12,2)),
    [0,2],
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/OpenAI/gpt2_128k",
)
# %%

names = [
    model.transformer.h[i]._module_path
    for i in range(0,12,2)
]

with open("/share/u/caden/sae-auto-interp/scripts/unique.json") as f:
    data = json.load(f)

module_filter = {name:torch.tensor(data[name], device="cuda:0") for name in names}

cache = FeatureCache(
    model, 
    submodule_dict, 
    filters=module_filter
)

from sae_auto_interp.utils import load_tokenized_data

tokens = load_tokenized_data(model.tokenizer)

cache.run(tokens, n_tokens=100_000)

cache.save_splits(n_splits=2, save_dir="/share/u/caden/sae-auto-interp/raw_features")