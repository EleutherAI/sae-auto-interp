# %%

import os
os.environ["CONFIG_PATH"] = "configs/caden_gpt2.yaml"

from sae_auto_interp.autoencoders import load_autoencoders
from nnsight import LanguageModel

model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map="auto", dispatch=True)

submodule_dict = load_autoencoders(
    model,
    ae_layers=[0],
    weight_dir="/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/Sam/pythia-70m-deduped",
)

# %%

DATA = "HELLO"

a = exec("DATA")

print(a)