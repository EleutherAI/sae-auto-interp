# %%

import pickle as pkl 

with open('sparse.pkl', 'rb') as f:
    sparse = pkl.load(f)

sparse

# %%

all_data = []

for layer, features in sparse[2650].items():

    for feature, n_unique in features.items():

        data = (layer, feature, n_unique)
        all_data.append(data)

sorted_data = sorted(all_data, key=lambda x: x[2], reverse=False)

# %%

from sae_auto_interp.features.