# %%


import os
os.environ["CONFIG_PATH"] = "configs/caden_gpt2.yaml"

from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features.frequency import FrequencyCache
from nnsight import LanguageModel
import torch

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
print("Model loaded")

ae_dict, submodule_dict = load_autoencoders(
    model, 
    [2],
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2",
)
print("Autoencoders loaded")


# %% 

with model.trace("frequency"):
    cache = FrequencyCache(model, submodule_dict)
    cache.run()

# %%