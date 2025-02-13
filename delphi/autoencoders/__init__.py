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
            model,
            run_cfg.sparse_model,
            run_cfg.hookpoints,
        )
    else:
        # I think this is better if we want to support gemmascope autoencoders
        # model path will always be of the form google/gemma-scope-<size>-pt-<type>/
        # where <size> is the size of the model and <type> is either res or mlp
        model_path = "google/" + run_cfg.sparse_model.split("/")[1]
        type = model_path.split("-")[-1]
        # we can use the hookpoints to determine the layer, size and l0,
        # because the module is determined by the model name
        # the hookpoint should be in the format
        # layer_<layer>/width_<size>/average_l0_<l0>
        layers = []
        l0s = []
        sizes = []
        for hookpoint in run_cfg.hookpoints:
            layer = int(hookpoint.split("/")[0].split("_")[1])
            size = hookpoint.split("/")[1].split("_")[1]
            l0 = int(hookpoint.split("/")[2].split("_")[2])
            layers.append(layer)
            sizes.append(size)
            l0s.append(l0)

        hookpoint_to_sae_encode = load_gemma_autoencoders(
            model_path=model_path,
            ae_layers=layers,
            average_l0s=l0s,
            sizes=sizes,
            type=type,
            dtype=model.dtype,
            device=model.device,
        )

    return hookpoint_to_sae_encode
