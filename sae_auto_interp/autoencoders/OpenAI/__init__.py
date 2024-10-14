from functools import partial
from pathlib import Path
from typing import List, Any, Callable, Dict, Tuple

import torch

from ..wrapper import AutoencoderLatents
from .model import ACTIVATIONS_CLASSES, Autoencoder


DEVICE = "cuda:0"


def load_oai_autoencoders(model, ae_layers: list[int], weight_dir: str):
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

        submodule.ae = AutoencoderLatents(ae, partial(_forward, ae), width=131_072)

        submodules[submodule._module_path] = submodule

    with model.edit(" "):
        for _, submodule in submodules.items():
            acts = submodule.output[0]
            submodule.ae(acts, hook=True)

    return submodules


def load_random_oai_autoencoders(
    model: Any,
    ae_layers: List[int],
    n_latents: int,
    k: int,
    seed: int = 42,
    save_dir: str | None = None,
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
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    for layer in ae_layers:
        submodule = model.model.layers[layer]
        sae = Autoencoder(n_latents, submodule.mlp.gate_proj.in_features, activation=ACTIVATIONS_CLASSES["TopK"](k=k), normalize=False, tied=False)
        sae.to(DEVICE).to(model.dtype)
        # Randomize the weights
        sae.encoder.weight.data.normal_(0, 1, generator=generator)
        sae.encoder.weight.data = sae.encoder.weight.data / torch.norm(sae.encoder.weight.data, dim=1, keepdim=True)
        sae.decoder.weight.data = sae.encoder.weight.data.T.clone()

        if save_dir is not None:
            save_path = Path(save_dir) / f"layer_{layer}/width_{n_latents}/k_{k}_seed_{seed}/params.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sae.state_dict(), save_path)
        
        def _forward(ae, x):
            return ae.encode(x)[0]
        
        submodule.ae = AutoencoderLatents(
            sae, partial(_forward, sae), width=n_latents
        )

        submodules[submodule.path] = submodule

    with model.edit("") as edited:
        for path, submodule in submodules.items():
            acts = submodule.output[0]
            submodule.ae(acts, hook=True)

    return submodules, edited