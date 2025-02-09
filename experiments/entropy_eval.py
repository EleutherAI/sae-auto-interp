#%%
from argparse import Namespace
from delphi.config import ExperimentConfig, FeatureConfig
from delphi.features import FeatureDataset, FeatureLoader
from delphi.features.constructors import default_constructor
from delphi.features.samplers import sample
import torch
import json
args = Namespace(
    module=".model.layers.16.router",
    experiment_options=ExperimentConfig(),
    feature_options=FeatureConfig(),
    features=100,
    model="monet_cache_converted/850m"
)
module = args.module
feature_cfg = args.feature_options
experiment_cfg = args.experiment_options
n_features = args.features  
start_feature = 0
sae_model = args.model

raw_dir = f"results/{args.model}"
features = torch.arange(start_feature,start_feature+n_features)
feature_dict = {f"{module}": features}

dataset = FeatureDataset(
    raw_dir=raw_dir,
    cfg=feature_cfg,
    modules=[module],
    features=feature_dict,
)
# %%
def set_record_buffer(record, buffer_output):
    record.buffer = buffer_output
loader = FeatureLoader(dataset, constructor=set_record_buffer, sampler=lambda x: x, transform=lambda x: x)
entropies = []
for record in loader:
    buffer = record.buffer
    locations, activations, tokens = buffer.locations, buffer.activations, buffer.tokens
    feature_id = record.feature.feature_index
    all_tokens = tokens[locations[:, 0], locations[:, 1]]
    _, token_counts = torch.unique(all_tokens, return_counts=True, sorted=False)
    token_counts = token_counts.double()
    token_probs = token_counts / token_counts.sum()
    token_entropy = -(token_probs * token_probs.log2()).sum()
    entropies.append(token_entropy.item())
# %%
import matplotlib.pyplot as plt
plt.hist(entropies, bins=20)
plt.xlabel("Entropy")
plt.ylabel("Number of features")
plt.title("Token entropy distribution")
plt.show()
# %%