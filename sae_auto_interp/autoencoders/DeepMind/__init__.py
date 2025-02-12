from functools import partial
from .model import JumpReLUSAE
from typing import List, Dict
import torch
from ..wrapper import AutoencoderLatents
DEVICE = "cuda:0"




def load_gemma_autoencoders(model, ae_layers: list[int],average_l0s: Dict[int,int],size:str,type:str):
    submodules = {}

    for layer in ae_layers:
    
        path = f"layer_{layer}/width_{size}/average_l0_{average_l0s[layer]}"
        sae = JumpReLUSAE.from_pretrained(path,type,"cuda")
        
        #sae.half()
        def _forward(sae, x):
            encoded = sae.encode(x)
            return encoded
        if type == "res":
            submodule = model.model.layers[layer]
        elif type == "mlp":
            submodule = model.model.layers[layer].post_feedforward_layernorm
        submodule.ae = AutoencoderLatents(
            sae, partial(_forward, sae), width=sae.W_enc.shape[1]
        )

        submodules[submodule.path] = submodule

    with model.edit(" ") as edited:
        for _, submodule in submodules.items():
            if type == "res":
                acts = submodule.output[0]
            else:
                acts = submodule.output
            submodule.ae(acts, hook=True)

    return submodules, edited
