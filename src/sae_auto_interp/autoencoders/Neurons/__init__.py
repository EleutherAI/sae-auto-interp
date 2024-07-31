from typing import List
import torch
from functools import partial
from ..OpenAI.model import ACTIVATIONS_CLASSES, TopK

DEVICE = "cuda:0"

class TopKNeurons(torch.nn.Module):
    def __init__(self, k:int):
        super().__init__()
        self.k = k

    def forward(self, x):
        topk = TopK(self.k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
        return topk(x)


def load_llama3_neurons(
        model,
        layers:List[int],
        k:int
    ):
    submodule_dict = {}
    for layer in layers:
        submodule = model.model.layers[layer].mlp.down_proj

        submodule.ae = TopKNeurons(k)
        submodule.ae.width = submodule.in_features
        submodule_dict[layer] = submodule
    
    with model.edit(" "):
        for _, submodule in submodule_dict.items():
            acts = submodule.input[0][0]
            submodule.ae(acts, hook=True)


    return submodule_dict


