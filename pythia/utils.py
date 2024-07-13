import json
import torch

def load_module_stuff(path, modules=["embed", "attn", "mlp", "resid"], device="cuda:0"):

    module_map = {
        "embed" : ".gpt_neox.embed_in",
        "mlp" : ".gpt_neox.layers.%d.mlp",
        "attn" : ".gpt_neox.layers.%d.attention",
        "resid" : ".gpt_neox.layers.%d"
    }

    with open(path, 'r') as f:
        data = json.load(f) 

    to_save = {}

    if "embed" in modules:
        to_save[module_map["embed"]] = torch.tensor(data["embed"], device=device)
    
    data.pop("embed")

    for name, data in data.items():
        if data == []:
            continue

        module, layer = name.split('_')

        if module in modules:
            to_save[module_map[module] % int(layer)] = torch.tensor(data, device=device)

    return to_save