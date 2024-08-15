from functools import partial
from .model import JumpReLUSAE
from typing import List, Dict

from ..wrapper import AutoencoderLatents
DEVICE = "cuda:0"




def load_gemma_autoencoders(model, ae_layers: List[int],average_l0s: Dict[int,int]):
    submodules = {}

    for layer in ae_layers:
        path = f"layer_{layer}/width_16k/average_l0_{average_l0s[layer]}"

        sae = JumpReLUSAE.from_pretrained(path)
        sae.to(DEVICE)
        def _forward(sae, x):
            encoded = sae.encode(x)
            return encoded

        submodule = model.model.layers[layer]
        submodule.ae = AutoencoderLatents(
            sae, partial(_forward, sae), width=16384
        )

        submodules[layer] = submodule

    with model.edit(" "):
        for _, submodule in submodules.items():
            acts = submodule.output[0]
            submodule.ae(acts, hook=True)

    return submodules
