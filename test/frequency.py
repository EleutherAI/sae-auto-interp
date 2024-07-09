import os
os.environ["CONFIG_PATH"] = "configs/frequency_cache.yaml"

import torch
from nnsight import LanguageModel
import matplotlib.pyplot as plt

from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features.frequency import FrequencyCache
from sae_auto_interp.utils import load_tokenized_data

LAYER = 0
N_BATCHES = 20

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
print("Model loaded")

ae_dict, submodule_dict = load_autoencoders(
    model, 
    [LAYER],
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2",
)
print("Autoencoders loaded")

cache = FrequencyCache(model, submodule_dict)
cache.run()
results = cache.save()


tokens = load_tokenized_data(model.tokenizer, batch_len=1024)
tokens = tokens[:N_BATCHES]

data = {}

for layer, selected_features in results.items():
    with model.trace(tokens):

        ae_out = model.transformer.h[0].ae.output
        ae_out = torch.mean(ae_out, dim=0)
        ae_out = ae_out[:,selected_features]
        ae_out.save()

    data[layer] = ae_out.value.detach().cpu().numpy()

plt.plot(data[0])
plt.savefig("graphs/plots/frequency.png")
