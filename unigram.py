# %%

import os
os.environ["CONFIG_PATH"] = "configs/caden_gpt2.yaml"

from nnsight import LanguageModel 
from tqdm import tqdm
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.autoencoders import load_autoencoders
from sae_auto_interp.features import (
    CombinedStat, Feature, FeatureRecord
)
import torch
from sae_auto_interp.features.stats import Activation, Logits
from collections import defaultdict
from tqdm   import tqdm

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
submodule_dict = load_autoencoders(
    model, 
    list(range(0,12,2)),
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/OpenAI/gpt2"
)

tokens = load_tokenized_data(model.tokenizer)

# %%


raw_features_path = "/share/u/caden/sae-auto-interp/raw_features"
processed_features_path = "/share/u/caden/sae-auto-interp/feature_statistics"


k_values = torch.arange(50, 1050, 100)

results = defaultdict(lambda: defaultdict(list))

lemmatized_results = defaultdict(lambda: defaultdict(list))

for layer, submodule in submodule_dict.items():
    ae = submodule.ae._module

    records = FeatureRecord.from_tensor(
        tokens,
        layer_index=layer,
        tokenizer=model.tokenizer,
        raw_dir=raw_features_path,
        selected_features=range(1000),
        min_examples=50,
        max_examples=5000,
    )

    stat = CombinedStat(
        activation = Activation(
            k=20,
            get_lemmas=True
        )
    )
    
    for k in tqdm(k_values):
        stat.refresh(k=k)

        stat.compute(records)
        
        for record in records:
            if len(record.examples) < k:
                continue

            n_unique = record.unique_tokens
            results[layer][k.item()].append(n_unique)

            lemmatized_results[layer][k.item()].append(record.n_lemmas)

# %%

import numpy as np
import matplotlib.pyplot as plt

# Sample plotting function
def plot_results(results, lemmatized_results, save_path='plots/combined_plots.png'):
    num_layers = len(results)
    
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 6 * num_layers))
    fig.subplots_adjust(hspace=0.4)
    
    if num_layers == 1:
        axes = [axes]
    
    for i, layer in enumerate(results.keys()):
        ks = sorted(results[layer].keys())
        print(ks)

        n_unique = [results[layer][k] for k in ks]
        n_lemmas = [lemmatized_results[layer][k] for k in ks]
        
        means_unique = [np.mean(acts) for acts in n_unique]
        std_devs_unique = [np.std(acts) for acts in n_unique]

        means_lemmas = [np.mean(acts) for acts in n_lemmas]
        std_devs_lemmas = [np.std(acts) for acts in n_lemmas]
        
        # Scatter plot with error bands
        ax = axes[i]
        
        # Scatter plot for n_unique
        for k, acts in zip(ks, n_unique):
            ax.scatter([k]*len(acts), acts, label=f'n_unique k={k}' if k == ks[0] else "", color='blue', alpha=0.6)
        
        # Error bands for n_unique
        ax.plot(ks, means_unique, label='Mean n_unique', color='blue')
        ax.fill_between(ks, np.array(means_unique) - np.array(std_devs_unique), np.array(means_unique) + np.array(std_devs_unique), color='blue', alpha=0.2)
        
        # Scatter plot for n_lemmas
        for k, acts in zip(ks, n_lemmas):
            ax.scatter([k]*len(acts), acts, label=f'n_lemmas k={k}' if k == ks[0] else "", color='orange', alpha=0.6)
        
        # Error bands for n_lemmas
        ax.plot(ks, means_lemmas, label='Mean n_lemmas', color='orange')
        ax.fill_between(ks, np.array(means_lemmas) - np.array(std_devs_lemmas), np.array(means_lemmas) + np.array(std_devs_lemmas), color='orange', alpha=0.2)
        
        ax.set_title(f'Scatter Plot with Error Bands for {layer}')
        ax.set_xlabel('k')
        ax.set_ylabel('Values')
        ax.legend()
    
    # Save the combined figure
    plt.savefig(save_path)
    plt.show()

plot_results(results, lemmatized_results)
