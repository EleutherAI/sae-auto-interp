from functools import partial
from typing import List, Any, Tuple, Optional, Dict
import torch
import torch.nn as nn
from sae import Sae, SaeConfig
from pathlib import Path
from .OpenAI.model import ACTIVATIONS_CLASSES, TopK
from .wrapper import AutoencoderLatents

DEVICE = "cuda:0"


def load_eai_autoencoders(
    model: Any,
    ae_layers: List[int],
    weight_dir: str,
    module: str,
    randomize: bool = False,
    seed: int = 42,
    k: Optional[int] = None
) -> Tuple[Dict[str, Any], Any]:
    """
    Load EleutherAI autoencoders for specified layers and module.

    Args:
        model (Any): The model to load autoencoders for.
        ae_layers (List[int]): List of layer indices to load autoencoders for.
        weight_dir (str): Directory containing the autoencoder weights.
        module (str): Module name ('mlp' or 'res').
        randomize (bool, optional): Whether to randomize the autoencoder. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        k (Optional[int], optional): Number of top activations to keep. Defaults to None.

    Returns:
        Tuple[Dict[str, Any], Any]: A tuple containing the submodules dictionary and the edited model.
    """
    submodules = {}

    for layer in ae_layers:
        if module=="mlp":
            submodule = f"layers.{layer}.{module}"
        elif module=="res":
            submodule = f"layers.{layer}"
        
        if "mnt" in weight_dir:
            sae = Sae.load_from_disk(weight_dir+"/"+submodule,device=DEVICE).to(dtype=model.dtype)
        else:
            sae = Sae.load_from_hub(weight_dir,hookpoint=submodule, device=DEVICE).to(dtype=model.dtype)
        
        if randomize:
            sae = Sae.load_from_hub(weight_dir,hookpoint=submodule, device=DEVICE).to(dtype=model.dtype)
            sae = Sae(sae.d_in, sae.cfg, device=DEVICE, dtype=model.dtype, decoder=False)
            # Randomize the weights
            sae.encoder.weight.data.normal_(-1,1)
            sae.encoder.weight.data = sae.encoder.weight.data / torch.norm(sae.encoder.weight.data, dim=0, keepdim=True)
            sae.W_dec = sae.encoder.weight.data.T
        
        def _forward(sae, k,x):
            encoded = sae.pre_acts(x)
            if k is not None:
                trained_k = k
            else:
                trained_k = sae.cfg.k
            topk = TopK(trained_k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
            return topk(encoded)

        if "llama" in weight_dir:
            if module == "res":
                submodule = model.model.layers[layer]
            else:
                submodule = model.model.layers[layer].mlp
        elif "gpt2" in weight_dir:
            submodule = model.transformer.h[layer]
        else:
            if module == "res":
                submodule = model.gpt_neox.layers[layer]
            else:
                submodule = model.gpt_neox.layers[layer].mlp
        submodule.ae = AutoencoderLatents(
            sae, partial(_forward, sae, k), width=sae.encoder.weight.shape[0]
        )

        submodules[submodule.path] = submodule

    with model.edit("") as edited:
        for path, submodule in submodules.items():
            if "embed" not in path and "mlp" not in path:
                acts = submodule.output[0]
            else:
                acts = submodule.output
            submodule.ae(acts, hook=True)

    return submodules,edited
