import os
os.environ["CONFIG_PATH"] = "/mnt/ssd-1/gpaulo/SAE-Zoology/sae_auto_interp/configs/gpaulo_llama_256.yaml"

from nnsight import LanguageModel 
from tqdm import tqdm
import torch
import argparse
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import CombinedStat, Logits, FeatureRecord, Activation

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
# Load tokenized data
n_tokens = 10_400_000
dataset_repo =  "kh4dien/fineweb-100m-sample"
batch_len = 256
tokens  = load_tokenized_data(model.tokenizer,n_tokens=n_tokens, dataset_repo=dataset_repo, batch_len=batch_len,dataset_split="train")

# Load features I want to explain
print("Samples loaded")
raw_features_path = "raw_features_llama"
processed_features_path = "processed_features_llama"

# You can add any object that inherits from Stat
# to combined stats. This info is added to the record
stats = CombinedStat(
    # logits = Logits(
    #     tokenizer=model.tokenizer,
    #     W_U = model.model.norm.weight * model.lm_head.weight,
    #     k=10,
    # ),
    activations = Activation(
        k=1000,
    ),
)    

for layer, _ in submodule_dict.items():

    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        raw_features_path,
        selected_features=torch.arange(0,50),
        max_examples=10000
    )
    # Refresh updates a memory intensive caches for stuff like
    # umap locations or logit matrices
    # W_dec = model.ae.autoencoder.W_dec.T
    # stats.refresh(W_dec=W_dec.bfloat16())
    # Compute updates records with stat information
    stats.compute(records,save_dir=processed_features_path,tokenizer=model.tokenizer)
