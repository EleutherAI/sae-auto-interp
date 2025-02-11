from functools import partial, reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from nnsight import LanguageModel
from sparsify import Sae

from .Custom.openai import ACTIVATIONS_CLASSES, TopK
from .wrapper import AutoencoderLatents

DEVICE = "cuda:0"


def load_eai_autoencoders(
    model: Any,
    ae_layers: List[int],
    weight_dir: str,
    module: str,
    transcoder: bool = False,
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
        if "pythia" in weight_dir:
            if module == "res":
                submodule = model.gpt_neox.layers[layer]
            else:
                submodule = model.gpt_neox.layers[layer].mlp

        elif "gpt2" in weight_dir:
            submodule = model.transformer.h[layer]
        else:
            if module == "res":
                submodule = model.model.layers[layer]
            else:
                submodule = model.model.layers[layer].mlp
            
        submodule.ae = AutoencoderLatents(
            sae, partial(_forward, sae, k), width=sae.encoder.weight.shape[0],hookpoint=submodule.path
        )

        submodules[submodule.path] = submodule

    with model.edit("") as edited:
        for path, submodule in submodules.items():
            if "embed" not in path and "mlp" not in path:
                if transcoder:
                    acts = submodule.input[0]
                else:
                    acts = submodule.output[0]
            else:
                if transcoder:
                    acts = submodule.input
                else:
                    acts = submodule.output
            submodule.ae(acts, hook=True)

    return submodules, edited
def resolve_path(model, path_segments: List[str]) -> List[str] | None:
    """Attempt to resolve the path segments to the model in the case where it has been wrapped 
    (e.g. by a LanguageModel, causal model, or classifier)."""
    # If the first segment is a valid attribute, return the path segments
    if hasattr(model, path_segments[0]):
        return path_segments

    # Look for the first actual model inside potential wrappers
    for attr_name, attr in model.named_children():
        if isinstance(attr, (torch.nn.Module, LanguageModel)):
            print(f"Checking wrapper model attribute: {attr_name}")
            if hasattr(attr, path_segments[0]):
                print(f"Found matching path in wrapper at {attr_name}.{path_segments[0]}")
                return [attr_name] + path_segments
            
            # Recursively check deeper
            deeper_path = resolve_path(attr, path_segments)
            if deeper_path is not None:
                print(f"Found deeper matching path starting with {attr_name}")
                return [attr_name] + deeper_path
    return None


def load_and_hook_sparsify_models(
    model: LanguageModel,
    name: str,
    hookpoints: List[str],
    k: Optional[int] = None,
    device: str | torch.device | None = None
) -> Tuple[Dict[str, Any], Any]:
    """
    Load sparsify autoencoders for specified hookpoints.

    Args:
        model (Any): The model to load autoencoders for.
        name (str): The name of the sparse model to load. If the model is on-disk this is the path
            to the directory containing the sparse model weights.
        hookpoints (List[str]): List of hookpoints to load autoencoders for.
        k (Optional[int], optional): Number of top activations to keep. Defaults to None.
        device (str | torch.device | None, optional): The device to load the sparse models on.
            If not specified the sparse models will be loaded on the same device as the base model.

    Returns:
        Tuple[Dict[str, Any], Any]: A tuple containing the submodules dictionary and the edited model.
    """
    if device is None:
        device = model.device

    # Load the sparse models
    hookpoint_to_sparse = {}
    name_path = Path(name)  
    if name_path.exists():
        for hookpoint in hookpoints:
            hookpoint_to_sparse[hookpoint] = Sae.load_from_disk(name_path / hookpoint, device=device)
    else:
        sparse_models = Sae.load_many(name, device=device)
        hookpoint_to_sparse.update({hookpoint: sparse_models[hookpoint] for hookpoint in hookpoints})
        del sparse_models

    # Add sparse models to submodules
    def forward_fn(sae, k, x):
        encoded = sae.pre_acts(x)
        trained_k = k if k is not None else sae.cfg.k
        return TopK(trained_k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())(encoded)

    submodules = {}
    for hookpoint, sparse_model in hookpoint_to_sparse.items():
        path_segments = resolve_path(model, hookpoint.split('.'))
        if path_segments is None:
            raise ValueError(f"Could not find valid path for hookpoint: {hookpoint}")
        
        submodule = reduce(getattr, path_segments, model)

        submodule.ae = AutoencoderLatents(
            sparse_model,
            partial(forward_fn, sparse_model, k),
            width=sparse_model.encoder.weight.shape[0]
        )
        submodules[hookpoint] = submodule

    # Edit base model to collect sparse model activations
    with model.edit("") as edited:
        for path, submodule in submodules.items():
            acts = submodule.output[0] if "embed" not in path and "mlp" not in path else submodule.output
            submodule.ae(acts, hook=True)

    return submodules, edited
