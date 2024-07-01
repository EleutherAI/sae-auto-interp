from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features.cache import FeatureCache
from sae_auto_interp.utils import get_samples
from nnsight import LanguageModel
import torch

# Load model and autoencoders
model = LanguageModel("meta-llama/Meta-Llama-3-8b", device_map="auto", dispatch=True,dtype=torch.bfloat16)
print("Model loaded")
# Load autoencoders, submodule dict, and edits.
# Submodule dict is used in caching to save ae latents
# Edits are applied to the model
ae_dict, submodule_dict, edits = load_autoencoders(
    model, 
    [0,2,4,6],
    "saved_autoencoders/Meta-Llama-3-8B/"
)
print("Autoencoders loaded")
# Set a default alteration on the model
with model.alter(" ", edits=edits):
    for layer_idx, _ in ae_dict.items():
        layer = model.transformer.h[layer_idx]
        acts = layer.output[0]
        layer.ae(acts, hook=True)

# Get and sort samples
samples = get_samples()
samples = {
    int(layer) : features
    for layer, features in samples.items() 
    if int(layer) in ae_dict.keys()
}
print("Samples loaded")
# Cache and save features
cache = FeatureCache(model, submodule_dict)
cache.run()
print("Caching complete")
for layer in [0,2,4,6]:
    feature_range = torch.tensor(samples[layer])
    cache.save_selected_features(feature_range, layer, save_dir="raw_features")
print("Selected features saved")