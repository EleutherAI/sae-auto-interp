# %%
import os
import json
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

directory = "scores/oai"

def load_data(directory):
    data = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            with open(os.path.join(directory, file), "r") as f:
                d = (
                    file,
                    json.load(f)['ev_correlation_score']
                )
                data.append(d)
    return data

# %%
# Load the data
data = load_data(directory)

# Process the data
layer_stats = defaultdict(list)

for file, score in data:
    layer = int(file.split("_")[0].split("layer")[-1])
    layer_stats[layer].append(score)

# Create histograms
fig, axs = plt.subplots(len(layer_stats), 1, figsize=(10, 5*len(layer_stats)), sharex=True)
fig.suptitle('Histograms of Scores by Layer')

for i, (layer, scores) in enumerate(sorted(layer_stats.items())):
    axs[i].hist(scores, bins=20, range=(0, 1), edgecolor='black')
    axs[i].set_title(f'Layer {layer}')
    axs[i].set_ylabel('Frequency')
    axs[i].set_xlim(0, 1)
    axs[i].set_xticks(np.arange(0, 1.1, 0.1))

axs[-1].set_xlabel('Score')

plt.tight_layout()
plt.show()