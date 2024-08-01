# %%
import pickle as pkl
import random
from collections import defaultdict

import torch
from nnsight import LanguageModel
from tqdm import tqdm

from sae_auto_interp.autoencoders import load_oai_autoencoders

model = LanguageModel("gpt2", device_map="auto", dispatch=True)
tokenizer = model.tokenizer
submodule_dict = load_oai_autoencoders(
    model=model, ae_layers=list(range(0, 12, 2)), weight_dir="weights/gpt2_128k"
)


with open("sparse_08.pkl", "rb") as f:
    data = pkl.load(f)


def gen(token, n=2):
    sentences = []

    for _ in range(n):
        tokens = random.sample(range(len(tokenizer.vocab)), 19)

        tokens.append(token)

        random.shuffle(tokens)

        sentences.append(tokens)

    return sentences


def _score(examples: list[int], submodule: str, feature: int):
    with model.trace(examples):
        scores = submodule.ae.output

        score = torch.any(scores[:, :, feature] != 0, dim=0)

        score = score.sum().item().save()

    return score.value


scores = defaultdict()

for module, features in data.items():
    for feature, unique_tokens in tqdm(features.items()):
        batch = []

        for unique in unique_tokens:
            batch.extend(gen(unique))

        score = _score(batch, submodule_dict[module], feature)

        f = f"{module}_{feature}.json"
        s = score / (len(unique_tokens) * 2)
        scores[f] = (score, len(unique_tokens), score / (len(unique_tokens) * 2))


# %%

fractions = defaultdict(dict)

for i in torch.arange(0.0, 1.0, 0.1):
    i = i.item()
    total_per_layer = {
        ".transformer.h.0": 802,
        ".transformer.h.2": 814,
        ".transformer.h.4": 893,
        ".transformer.h.6": 888,
        ".transformer.h.8": 856,
        ".transformer.h.10": 807,
    }

    n_per_layer = defaultdict(int)

    for feature, score in scores.items():
        layer = feature.split("_")[0]
        if score[2] > i:
            n_per_layer[layer] += 1

    for layer, n in n_per_layer.items():
        fractions[i][layer] = n / total_per_layer[layer]


# %%

lines = [list(f.values()) for f in fractions.values()]

# %%

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Assuming fractions is already defined as before

layers = [0, 2, 4, 6, 8, 10]
thresholds = torch.arange(0.0, 1.0, 0.1)

fig, ax = plt.subplots(figsize=(12, 8))

norm = Normalize(vmin=thresholds.min(), vmax=thresholds.max())
cmap = plt.cm.viridis

for threshold in thresholds:
    values = [
        fractions[threshold.item()][f".transformer.h.{layer}"] for layer in layers
    ]
    ax.plot(layers, values, marker="o", linestyle="-", color=cmap(norm(threshold)))

ax.set_title("Fraction of Unigram Features per Layer", fontsize=16)
ax.set_xlabel("Layers 0 to 10 in GPT-2", fontsize=14)
ax.set_ylabel("Fraction of Features with Context Independent Activations", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.7)
ax.set_xticks(layers)
ax.set_ylim(0, 1)

# Add color scale
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.05)
cbar.set_ticks(thresholds)
cbar.set_ticklabels([f"{t:.1f}" for t in thresholds])

plt.tight_layout()
plt.show()


# %%
#
import matplotlib.pyplot as plt
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Assuming fractions is already defined as before
thresholds = torch.arange(0.03, 0.06, 0.01)

fig, ax = plt.subplots(figsize=(12, 8))

vmin = thresholds.min().item()
vmax = thresholds.max().item()
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.viridis

# Example data for n_positional and layers
n_positional = {
    0.03: [0.1, 0.2, 0.3, 0.4, 0.5],
    0.04: [0.15, 0.25, 0.35, 0.45, 0.55],
    0.05: [0.2, 0.3, 0.4, 0.5, 0.6],
}
layers = range(5)

for threshold, values in n_positional.items():
    print(cmap(norm(threshold)))
    ax.plot(values, marker="o", linestyle="-", color=cmap(norm(threshold)))


ax.set_title("Fraction of Unigram Features per Layer", fontsize=16)
ax.set_xlabel("Layers 0 to 10 in GPT-2", fontsize=14)
ax.set_ylabel("Fraction of Features with Context Independent Activations", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.7)
ax.set_xticks(layers)

# Add color scale
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.05)
cbar.set_ticks(thresholds.numpy())
cbar.set_ticklabels([f"{t:.2f}" for t in thresholds])

plt.tight_layout()
plt.show()
