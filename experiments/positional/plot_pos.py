# %%

import json
import os
from collections import defaultdict

directories = [f"./pos_0{n}" for n in range(1, 7)]

n_positional = defaultdict(list)

for i, directory in enumerate(directories):
    n = i + 1
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            path = os.path.join(directory, file)
            with open(path, "r") as f:
                data = json.load(f)

            n_positional[n * 0.01].append(len(data))

print(n_positional)

# %%
import matplotlib.pyplot as plt
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Assuming fractions is already defined as before
thresholds = torch.arange(0.01, 0.07, 0.01)

fig, ax = plt.subplots(figsize=(12, 8))

vmin = thresholds.min().item()
vmax = thresholds.max().item()
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.viridis

# Example data for n_positional and layers
# n_positional = {
#     0.03: [0.1, 0.2, 0.3, 0.4, 0.5],
#     0.04: [0.15, 0.25, 0.35, 0.45, 0.55],
#     0.05: [0.2, 0.3, 0.4, 0.5, 0.6]
# }
# layers = range(5)

for threshold, values in n_positional.items():
    print(threshold)
    print(cmap(norm(threshold)))
    ax.plot(values, marker="o", linestyle="-", color=cmap(norm(threshold)))


ax.set_title("Number of positional features per layer", fontsize=16)
ax.set_xlabel("Layer in GPT-2", fontsize=14)
ax.set_ylabel("Number of positional features", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.7)
ax.set_xticks(range(12))

# Add color scale
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.05)
cbar.set_ticks(thresholds.numpy())
cbar.set_ticklabels([f"{t:.2f}" for t in thresholds])

plt.tight_layout()
plt.show()
