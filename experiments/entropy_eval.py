#%%
import sys
sys.path.append("..")
from tqdm.auto import tqdm
from argparse import Namespace
from delphi.config import ExperimentConfig, FeatureConfig
from delphi.features import FeatureDataset, FeatureLoader
from delphi.features.constructors import default_constructor
from delphi.features.samplers import sample
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from pathlib import Path
import numpy as np
import json
import os
from matplotlib import pyplot as plt
import seaborn as sns
import traceback
#%%
def compute_entropy(cache_dir, module):
    feature_cfg = FeatureConfig()
    experiment_cfg = ExperimentConfig()
    n_features = 1000
    start_feature = 0
    
    max_feat = 0
    n_splits = 0
    for st_file in (Path(cache_dir) / module).glob(f"*.safetensors"):
        start, end = map(int, st_file.stem.split("_"))
        max_feat = max(max_feat, end)
        n_splits += 1
    feature_cfg.width = max_feat + 1
    feature_cfg.n_splits = n_splits

    features = torch.arange(start_feature,start_feature+n_features)
    feature_dict = {f"{module}": features}

    dataset = FeatureDataset(
        raw_dir=cache_dir,
        cfg=feature_cfg,
        modules=[module],
        features=feature_dict,
    )
    def set_record_buffer(record, buffer_output):
        record.buffer = buffer_output
    loader = FeatureLoader(dataset, constructor=set_record_buffer, sampler=lambda x: x, transform=lambda x: x)
    entropies = []
    # for record in tqdm(loader, total=n_features):
    for record in loader:
        buffer = record.buffer
        locations, activations, tokens = buffer.locations, buffer.activations, buffer.tokens
        if tokens is None:
            tokens = dataset.load_tokens()
        feature_id = record.feature.feature_index
        all_tokens = tokens[locations[:, 0], locations[:, 1]]
        _, token_counts = torch.unique(all_tokens, return_counts=True, sorted=False)
        token_counts = token_counts.double()
        token_probs = token_counts / token_counts.sum()
        token_entropy = -(token_probs * token_probs.log2()).sum()
        entropies.append(token_entropy.item())
    return entropies
#%%
caches_dir = Path("/mnt/ssd-1/gpaulo/finished_articles/transcoders_beat_saes/raw_features")
model_info = {
    "pythia410m": dict(layer=16, layers=list(range(12, 22))),
    "pythia160m": dict(layer=10, layers=list(range(6, 16))),
    "llama": dict(layer=22, layers=list(range(12, 28))),
    "gemma": dict(layer=22, layers=list(range(12, 28))),
}
plots_dir = Path("results/entropy_plots")
for model_dir in caches_dir.glob("*"):
    model_name = model_dir.name
    if model_name not in model_info:
        continue
    model_plots_dir = plots_dir / model_name
    model_plots_dir.mkdir(exist_ok=True, parents=True)
    type_layer_entropies = defaultdict(dict)
    across_type_entropies = {}
    for sae_type_dir in tqdm(list(model_dir.glob("*"))):
        sae_type = sae_type_dir.name
        for layer in tqdm(model_info[model_name]["layers"]):
            prefix = "model" if "pythia" not in model_name else "gpt_neox"
            module_name = f".{prefix}.layers.{layer}.mlp"
            layer_dir = sae_type_dir / module_name
            if not layer_dir.exists():
                continue
            try:
                entropy = compute_entropy(sae_type_dir, module_name)
            except Exception as e:
                traceback.print_exc()
                continue
            type_layer_entropies[sae_type][layer] = entropy
            if layer == model_info[model_name]["layer"]:
                across_type_entropies[sae_type] = entropy
    sns.set_theme()
    for sae_type, entropies in type_layer_entropies.items():
        plt.plot(entropies.keys(), list(map(np.mean, entropies.values())), label=sae_type)
    plt.legend()
    plt.xlabel("Layer")
    plt.title(f"{model_name} Entropy by Layer")
    plt.savefig(model_plots_dir / "type_layer.png")
    plt.show()
    plt.clf()
    for sae_type, entropy in across_type_entropies.items():
        sns.kdeplot(entropy, bw_method=0.25, label=sae_type)
    plt.legend()
    plt.xlabel("Entropy")
    plt.title(f"{model_name} Entropy at Layer {model_info[model_name]['layer']}")
    plt.savefig(model_plots_dir / "across_type.png")
    plt.show()
# %%
