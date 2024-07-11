from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features.cache import FeatureCache
from nnsight import LanguageModel
import torch

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
print("Model loaded")

ae_dict, submodule_dict = load_autoencoders(
    model, 
    list(range(0,12,2)),
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2",
)
print("Autoencoders loaded")


# Cache and save features
cache = FeatureCache(model, submodule_dict)
cache.run()

save_dir = "raw_features"

print("Caching complete")
for layer in range(0,12,2):
    feature_range = torch.arange(0,1000)

    cache.save_selected_features(feature_range, layer, save_dir=save_dir)
print("Selected features saved")

