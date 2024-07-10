from nnsight import LanguageModel 
from tqdm import tqdm
import torch
import argparse
from sae_auto_interp.utils import load_tokenized_data, get_samples
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import CombinedStat, Logits, Feature, FeatureRecord

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]


# Load model and autoencoders
model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="auto", dispatch=True,torch_dtype =torch.bfloat16)
print("Model loaded")
# Load autoencoders, submodule dict
# Submodule dict is used in caching to save ae latents
# Edits are applied to the model
ae_dict, submodule_dict = load_autoencoders(
    model, 
    layers,
    "saved_autoencoders/llama-exp32"
)
print("Autoencoders loaded")
# Load tokenized data
tokens = load_tokenized_data(model.tokenizer)

# Load features I want to explain
samples = get_samples(N_LAYERS=32,N_FEATURES=131072)
features = Feature.from_dict(samples)
print("Samples loaded")
raw_features_path = "raw_features_llama"
processed_features_path = "processed_features_llama"

# You can add any object that inherits from Stat
# to combined stats. This info is added to the record
stats = CombinedStat(
    logits = Logits(
        model=model,
        k=10,
        get_top_logits=True
    )
)    

for layer, ae in ae_dict.items():

    selected_features = samples[layer]

    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        raw_features_path,
        model.tokenizer,
        selected_features=selected_features,
        max_examples=2000
    )
    # Refresh updates a memory intensive caches for stuff like
    # umap locations or logit matrices
    W_dec = ae.W_dec
    W_U = model.model.norm.weight * model.lm_head.weight
    stats.refresh(W_dec=W_dec.bfloat16(),W_U=W_U)
    # Compute updates records with stat information
    stats.compute(records)

    # Save the processed information to the processed feature dir
    for record in records:
        record.save(processed_features_path)