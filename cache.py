# %%
from nnsight import LanguageModel
import os
os.environ["CONFIG_PATH"] = "configs/pythia.yaml"
from sae_auto_interp.autoencoders import load_autoencoders
from sae_auto_interp.features import FeatureCache

model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map="auto", dispatch=True)

submodule_dict = load_autoencoders(
    model,
    None,
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/Sam/pythia-70m-deduped",
    modules=["embed", "mlp"]
)

cache = FeatureCache(model, submodule_dict)
cache.run()

# %%

# save_dir = "raw_features"

# # Save the first 1000 features per layer
# for layer in range(0,12,2):
#     feature_range = torch.arange(0,1000)
#     cache.save_selected_features(feature_range, layer, save_dir=save_dir)

features = [28533, 29476, 31461, 31467, 32081, 32469]
import torch
cache.save_selected_features(torch.tensor(features), '.gpt_neox.embed_in', save_dir="raw_features")


# %%

