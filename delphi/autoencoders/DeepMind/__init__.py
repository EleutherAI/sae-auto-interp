from functools import partial

import torch

from ..Custom.gemmascope import JumpReluSae


def load_gemma_autoencoders(
    ae_layers: list[int],
    average_l0s: dict[int, int],
    size: str,
    type: str,
    dtype: torch.dtype = torch.bfloat16,
):
    submodules = {}

    for layer in ae_layers:
        path = f"layer_{layer}/width_{size}/average_l0_{average_l0s[layer]}"
        sae = JumpReluSae.from_pretrained(
            f"google/gemma-scope-9b-pt-{type}", path, "cuda"
        )

        sae.to(dtype)

        def _forward(sae, x):
            encoded = sae.encode(x)
            return encoded

        assert type in [
            "res",
            "mlp",
        ], "Only res and mlp are supported for gemma autoencoders"
        hookpoint = (
            f"layers.{layer}"
            if type == "res"
            else f"layers.{layer}.post_feedforward_layernorm"
        )

        submodules[hookpoint] = partial(_forward, sae)

    return submodules
