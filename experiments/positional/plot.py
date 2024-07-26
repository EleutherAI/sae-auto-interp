# %% 

import os
import json

n_positional = []

for file in os.listdir('.'):
    if file.endswith('.txt'):
    
        with open(file, 'r') as f:
            data = json.load(f)

        n_positional.append(len(data))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(n_positional, marker='o', linestyle='-', color='r')
plt.title('Number of Positional Elements in Text Files')
plt.xlabel('Layer Index')
plt.ylabel('Number of Positional Features')
plt.grid(True, linestyle='--', alpha=0.7)
plt.grid(True)
plt.show()
# %%
