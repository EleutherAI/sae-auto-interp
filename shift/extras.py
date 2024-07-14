"""
Dump for some scripts to get SFC nodes.
"""


running_nodes = None

import torch

node_threshold = 0.1

top_nodes = {}

for name, data in running_nodes.items():
    if name == "y":
        continue

    acts = data.act
    indices = torch.where(acts > node_threshold)[0]

    top_nodes[name] = indices.tolist()

top_nodes
    
import json 

# save the top nodes to new file

with open(f"sfc.json", 'w') as f:
    json.dump(top_nodes, f)