from .model import Autoencoder
from ..wrapper import AutoencoderLatents
from typing import List
import torch
from functools import partial

DEVICE = "cuda:0"

def load_oai_autoencoders(
    model, 
    ae_layers: List[int], 
    weight_dir: str
):

    submodules = {}

    for layer in ae_layers:
        path = f"{weight_dir}/{layer}.pt"
        state_dict = torch.load(path)
        ae = Autoencoder.from_state_dict(state_dict=state_dict)
        ae.to(DEVICE)

        def _forward(ae, x):
            latents, _ = ae.encode(x)
            return latents

        submodule = model.transformer.h[layer]
        submodule.ae = AutoencoderLatents(ae,partial(_forward, ae),n_features=131072)

        submodules[submodule._module_path] = submodule

    with model.edit(" "):
        for _, submodule in submodules.items():
            acts = submodule.output[0]
            submodule.ae(acts, hook=True)

    return submodules
