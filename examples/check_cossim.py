#%%
from glob import glob
import json
is_monet = False
if is_monet:
    model_size = "1.4b"
    prefix = f"../results/explanations/monet_cache_converted/{model_size}/default"
    layer = "16"
    n_features = 512 ** 2
    feature_descs = {}
    for feature in glob(f"{prefix}/.model.layers.{layer}.router_feature*.txt"):
        feature_idx = int(feature.split("feature")[1][:-4])
        feature_descs[feature_idx] = open(feature).read()[1:-1]
    desc = f"Monet {model_size.upper()} Layer {layer}"
else:
    prefix = f"../results/explanations/sae_pkm/with_pkm_transcoder/default"
    layer = "8"
    n_features = 50_000
    feature_descs = {}
    for feature in glob(f"{prefix}/gpt_neox.layers.8_feature*.txt"):
        feature_idx = int(feature.split("feature")[1][:-4])
        feature_descs[feature_idx] = open(feature).read()[1:-1]
    desc = f"PKM for Pythia 160M Layer {layer}"
# %%
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("NovaSearch/stella_en_400M_v5", trust_remote_code=True).cuda()
# st = SentenceTransformer("intfloat/e5-large-v2", trust_remote_code=True).cuda()
#%%
keys = list(feature_descs.values())
# pref = "passage: "
pref = ""
embeds = {k: v for k, v in
          zip(keys,
              st.encode([pref + k for k in keys]))}
#%%
from collections import defaultdict
import numpy as np
idces = list(feature_descs.keys())
embed_array = [embeds[feature_descs[i]] for i in idces]
embed_array = np.array(embed_array)
sims = st.similarity(embed_array, embed_array)
#%%
from math import ceil
root = int(ceil(n_features ** 0.5))
n_groups = n_features // root
idx_roots = np.floor(np.array(idces, dtype=np.float64) / float(root))
same_group = idx_roots[:, None] == idx_roots[None, :]
same = np.eye(sims.shape[0], dtype=np.bool_)
not_same = (sims * (~same & ~same_group)).topk(n_groups, dim=-1)[0][:, -1]
same = (sims * (~same & same_group)).max(-1)[0]
same = same[same != 0]
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()
bins = np.linspace(0, 1, 100)
plt.hist(same, bins=bins, alpha=0.5, label="Same group", density=True)
plt.hist(not_same, bins=bins, alpha=0.5, label="Not same group", density=True)
plt.ylabel("Density")
plt.xlabel("Cosine similarity")
plt.legend()
plt.title(f"{desc} feature description similarity")
# %%
