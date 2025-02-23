#%%
from glob import glob
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
sims = st.similarity(embed_array, embed_array).numpy()
from math import ceil
root = int(ceil(n_features ** 0.5))
n_features_padded = n_features + root - n_features % root
idx_roots = np.floor(np.array(idces, dtype=np.float64) / float(root))
same_group = idx_roots[:, None] == idx_roots[None, :]
same_latent = np.eye(sims.shape[0], dtype=np.bool_)
def pad_group(i):
    a = np.zeros((n_features_padded, n_features_padded), dtype=i.dtype)
    idx = np.array(idces)
    idx0 = idx[:, None] + idx[None, :] * 0
    idx1 = idx[:, None] * 0 + idx[None, :]
    a[idx0, idx1] = i
    a = a.reshape(n_features_padded, root, root)
    return a
sims, same_group, same_latent = map(pad_group, (sims, same_group, same_latent))
not_same = sims * (~same_latent & ~same_group)
not_same = not_same.max(-1)
n_samples = 16
if n_samples == 0:
    not_same = not_same.sum(-1) / (not_same != 0).sum(-1)
else:
    index = np.random.randint(0, root, size=(n_features_padded, n_samples))
    not_same = np.take_along_axis(not_same, index, axis=-1)
not_same = not_same.flatten()
not_same = not_same[not_same != 0]
same = sims * (~same_latent & same_group)
same = same.max(-1).flatten()
same = same[same != 0]
# not_same = (sims * (~same_latent & ~same_group)).topk(n_groups, dim=-1)[0][:, -1]
# same = (sims * (~same_latent & same_group)).max(-1)[0]
# same = same[same != 0]
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()
bins = np.linspace(0, 1, 25)
plt.hist(same, bins=bins, alpha=0.5, label="Same group", density=True)
plt.hist(not_same, bins=bins, alpha=0.5, label="Not same group", density=True)
plt.ylabel("Density")
plt.xlabel("Cosine similarity")
plt.legend()
plt.title(f"{desc} feature description similarity")
# %%
