from typing import Callable

from transformers import PreTrainedModel

from delphi.config import RunConfig

from .custom.gemmascope import load_gemma_autoencoders
from .load_sparsify import load_sparsify_autoencoders

__all__ = ["load_autoencoders"]


def load_autoencoders(
    model: PreTrainedModel,
    run_cfg: RunConfig,
) -> dict[str, Callable]:
    # Add SAE hooks to the model
    if "gemma" not in run_cfg.sparse_model:
        hookpoint_to_sae_encode = load_sparsify_autoencoders(
            model,  # type: ignore
            run_cfg.sparse_model,
            run_cfg.hookpoints,
        )
    else:
        # Doing a hack here to enable gemma autoencoders
        layers = [int(n.split(".")[1]) for n in run_cfg.hookpoints]

        first_hookpoint_len = len(run_cfg.hookpoints[0].split("."))

        type = "res" if first_hookpoint_len == 2 else "mlp"

        for hookpoint in run_cfg.hookpoints:
            assert (
                len(hookpoint.split(".")) == first_hookpoint_len
            ), "All hookpoints must be of the same type for Gemma SAEs"

        print(f"Loading {type} gemma SAEs for L0 = 47, W=131K...")

        hookpoint_to_sae_encode = load_gemma_autoencoders(
            ae_layers=layers,
            average_l0s={layer: 47 for layer in layers},
            size="131k",
            type=type,
            dtype=model.dtype,
        )

    return hookpoint_to_sae_encode
