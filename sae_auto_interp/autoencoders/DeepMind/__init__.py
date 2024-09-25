from functools import partial
from .model import JumpReLUSAE
from typing import List, Dict

from ..OpenAI.model import ACTIVATIONS_CLASSES, TopK
from ..wrapper import AutoencoderLatents
DEVICE = "cuda:0"


def load_gemma_autoencoders(model, ae_layers: list[int],average_l0s: Dict[int,int],size:str,type:str):
    submodules = {}

    for layer in ae_layers:
        path = f"layer_{layer}/width_{size}/average_l0_{average_l0s[layer]}"

        sae = JumpReLUSAE.from_pretrained(path,type)
        sae.to(DEVICE)
        sae.half()
        def _forward(sae, x):
            encoded = sae.encode(x)
            return encoded
        if type == "res":
            submodule = model.model.layers[layer]
        else:
            submodule = model.model.layers[layer].mlp
        
        submodule.ae = AutoencoderLatents(
            sae, partial(_forward, sae), width=sae.W_enc.shape[1]
        )

        submodules[submodule._module_path] = submodule

    with model.edit(" "):
        for _, submodule in submodules.items():
            if type == "res":
                acts = submodule.output[0]
            else:
                acts = submodule.output
            submodule.ae(acts, hook=True)

    return submodules


def load_gemma_topk_latents(model, ae_layers: list[int], k: int):
    submodules = {}

    for layer in ae_layers:
        
        topk = TopK(k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
        def _forward(x):
            return topk(x)
        
        submodule = model.model.layers[layer]
        
        submodule.ae = AutoencoderLatents(
            None, _forward, width=submodule.mlp.gate_proj.in_features
        )

        submodules[submodule.path] = submodule

    with model.edit() as edited:
        for _, submodule in submodules.items():
            acts = submodule.output[0]
            submodule.ae(acts, hook=True)

        return edited, submodules
