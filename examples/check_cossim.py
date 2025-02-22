#%%
from glob import glob
import json
prefix = "../results/explanations/monet_cache_converted/850m/default"
layer = "16"
feature_descs = {}
for feature in glob(f"{prefix}/.model.layers.{layer}.router_feature*.txt"):
    feature_idx = int(feature.split("feature")[1][:-4])
    feature_descs[feature_idx] = open(feature).read()[1:-1]
# %%
feature_descs
# %%
import os
os.listdir(".")