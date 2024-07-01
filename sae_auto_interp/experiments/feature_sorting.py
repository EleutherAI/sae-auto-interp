import sys

sys.path.append('..')

from sae_auto_interp.features.features import Feature, feature_loader
from sae_auto_interp.autoencoders.model import get_autoencoder
from sae_auto_interp.features.stats import CombinedStat, Logits, Activations

from nnsight import LanguageModel
from datasets import load_dataset

from transformer_lens import utils

import torch
import json

model = LanguageModel("gpt2", device_map="auto", dispatch=True)
data = load_dataset("stas/openwebtext-10k", split="train")

tokens = utils.tokenize_and_concatenate(
    data, 
    model.tokenizer, 
    max_length=64
)   

tokens = tokens.shuffle(22)['tokens']

with open("/share/u/caden/scoring/autointerp/scripts/samples.json", 'r') as f:
    samples = json.load(f)

autoencoders = {layer: get_autoencoder("gpt2", layer, "cuda", "/mnt/ssd-1/gpaulo/SAE-Zoology/saved_autoencoders/") for layer in [0,2,4,6,8,10]}

features = [
    Feature(
        layer_index=0,
        feature_index=sample,
    )
    for sample in samples["0"][:10]
]




stats = CombinedStat(
    logits = Logits(
        model=model,
        get_kurtosis=True,
        get_skewness=True,
        top_k_logits=10
    ),
    activations = Activations(
        get_lemmas=True,
        top_activating_k=100
    ),
)    

all_records = []
for ae, records in feature_loader(tokens, features, model, autoencoders):
    stats.refresh(W_dec=ae.decoder.weight)
    stats.add(records)
    all_records.extend(records)

stats.load(all_records)