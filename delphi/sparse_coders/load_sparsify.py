from functools import partial
from pathlib import Path
from typing import Callable

import torch
from sparsify import Sae
from torch import Tensor
from transformers import PreTrainedModel


def sae_dense_latents(sae: Sae, x: Tensor) -> Tensor:
    """Run `sae` on `x`, yielding the dense activations."""
    pre_acts = sae.pre_acts(x)
    acts, indices = sae.select_topk(pre_acts)
    return torch.zeros_like(pre_acts).scatter_(-1, indices, acts)


def resolve_path(model: PreTrainedModel, path_segments: list[str]) -> list[str] | None:
    """Attempt to resolve the path segments to the model in the case where it
    has been wrapped (e.g. by a LanguageModel, causal model, or classifier)."""
    # If the first segment is a valid attribute, return the path segments
    if hasattr(model, path_segments[0]):
        return path_segments

    # Look for the first actual model inside potential wrappers
    for attr_name, attr in model.named_children():
        if isinstance(attr, torch.nn.Module):
            print(f"Checking wrapper model attribute: {attr_name}")
            if hasattr(attr, path_segments[0]):
                print(
                    f"Found matching path in wrapper at {attr_name}.{path_segments[0]}"
                )
                return [attr_name] + path_segments

            # Recursively check deeper
            deeper_path = resolve_path(attr, path_segments)
            if deeper_path is not None:
                print(f"Found deeper matching path starting with {attr_name}")
                return [attr_name] + deeper_path
    return None


def load_sparsify_sparse_coders(
    model: PreTrainedModel,
    name: str,
    hookpoints: list[str],
    device: str | torch.device | None = None,
) -> dict[str, Callable]:
    """
    Load sparsify sparse coders for specified hookpoints.

    Args:
        model (Any): The model to load autoencoders for.
        name (str): The name of the sparse model to load. If the model is on-disk
            this is the path to the directory containing the sparse model weights.
        hookpoints (list[str]): list of hookpoints to load autoencoders for.
        device (str | torch.device | None, optional): The device to load the
            sparse models on. If not specified the sparse models will be loaded
            on the same device as the base model.

    Returns:
        tuple[dict[str, Any], Any]: A tuple containing the submodules dictionary
            and the edited model.
    """
    if device is None:
        device = model.device or "cpu"

    # Load the sparse models
    hookpoint_to_sparse = {}
    name_path = Path(name)
    if name_path.exists():
        for hookpoint in hookpoints:
            hookpoint_to_sparse[hookpoint] = Sae.load_from_disk(
                name_path / hookpoint, device=device
            )
    else:
        sparse_models = Sae.load_many(name, device=device)
        hookpoint_to_sparse.update(
            {hookpoint: sparse_models[hookpoint] for hookpoint in hookpoints}
        )
        del sparse_models

    submodules = {}
    for hookpoint, sparse_model in hookpoint_to_sparse.items():
        path_segments = resolve_path(model, hookpoint.split("."))
        if path_segments is None:
            raise ValueError(f"Could not find valid path for hookpoint: {hookpoint}")

        submodules[".".join(path_segments)] = partial(sae_dense_latents, sparse_model)

    return submodules
