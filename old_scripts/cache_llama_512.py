import os
os.environ["CONFIG_PATH"] = "/mnt/ssd-1/gpaulo/SAE-Zoology/sae_auto_interp/configs/gpaulo_llama_512.yaml"

from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features.cache import FeatureCache
from nnsight import LanguageModel
import torch
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]

# Load model and autoencoders
model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="auto", dispatch=True,torch_dtype =torch.bfloat16)
print("Model loaded")

submodule_dict = load_autoencoders(
    model, 
    layers,
    "saved_autoencoders/llama-exp32"
)

print("Autoencoders loaded")

print("Samples loaded")
# Cache and save features
cache = FeatureCache(model, submodule_dict)
cache.run()
print("Caching complete")
for layer in layers:
    feature_range = torch.arange(0,10000)
    cache.save_selected_features(feature_range, layer, save_dir="raw_features_llama_256")
print("Selected features saved")