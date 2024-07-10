import torch
from nnsight import LanguageModel

from sae_auto_interp.autoencoders import load_autoencoders
from sae_auto_interp.features import FeatureCache

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

submodule_dict = load_autoencoders(
    model, 
    list(range(0,12,2)),
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2",
)

cache = FeatureCache(model, submodule_dict)
cache.run()

save_dir = "raw_features"

# Save the first 1000 features per layer
for layer in range(0,12,2):
    feature_range = torch.arange(0,1000)
    cache.save_selected_features(feature_range, layer, save_dir=save_dir)

