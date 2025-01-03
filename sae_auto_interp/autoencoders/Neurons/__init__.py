from functools import partial
from typing import List

import torch

from ..Custom.openai import ACTIVATIONS_CLASSES, TopK

DEVICE = "cuda:0"

class TopKNeurons(torch.nn.Module):
    def __init__(self, k: int, input_dim: int, rotate: bool = False, seed: int = 42, device: str = DEVICE):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.rotate = rotate
        self.device = device
        # Set the seed for reproducibility
        torch.manual_seed(seed)
        
        if rotate:
            # Initialize the random rotation matrix
            self.rotation_matrix = torch.nn.init.orthogonal_(torch.empty(input_dim, input_dim, device=device))
        else:
            self.rotation_matrix = None

    def forward(self, x):
        if self.rotate:
            # Apply the random rotation
            x_rotated = x @ self.rotation_matrix 
        else:
            x_rotated = x
        
        # Apply TopK
        topk = TopK(self.k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
        return topk(x_rotated)


def load_llama3_neurons(
        model,
        layers: list[int],
        k: int,
        rotate: bool = False,
        seed: int = 42
    ):
    submodule_dict = {}
    for layer in layers:
        submodule = model.model.layers[layer].mlp.down_proj

        submodule.ae = TopKNeurons(k, input_dim=submodule.in_features, rotate=rotate, seed=seed,device=DEVICE)
        submodule.ae.width = submodule.in_features
        submodule_dict[layer] = submodule
    
    with model.edit(" ") as edited:
        for _, submodule in submodule_dict.items():
            acts = submodule.input
            submodule.ae(acts, hook=True)
    return submodule_dict,edited
