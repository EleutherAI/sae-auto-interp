from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features.cache import FeatureCache
from sae_auto_interp.utils import get_samples
from nnsight import LanguageModel
import torch
import argparse
from sae_auto_interp import cache_config as CONFIG

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]

# Load model and autoencoders
model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="auto", dispatch=True,torch_dtype =torch.bfloat16)
print("Model loaded")
# Load autoencoders, submodule dict, and edits.
# Submodule dict is used in caching to save ae latents
# Edits are applied to the model
ae_dict, submodule_dict = load_autoencoders(
    model, 
    layers,
    "saved_autoencoders/llama-exp32"
)

print("Autoencoders loaded")

# Get and sort samples
CONFIG.n_tokens = 10_400_000
CONFIG.dataset_repo =  "kh4dien/fineweb-100m-sample"
CONFIG.batch_len = 256

print("Samples loaded")
# Cache and save features
cache = FeatureCache(model, submodule_dict)
cache.run()
print("Caching complete")
for layer in layers:
    feature_range = torch.arange(0,10000)
    cache.save_selected_features(feature_range, layer, save_dir="raw_features_llama")
print("Selected features saved")