from functools import partial
from typing import List

from sae import Sae

from .OpenAI.model import ACTIVATIONS_CLASSES, TopK
from .wrapper import AutoencoderLatents

DEVICE = "cuda:0"


def load_eai_autoencoders(model, ae_layers: List[int], weight_dir: str):
    submodules = {}

    for layer in ae_layers:
        path = f"{weight_dir}/layers.{layer}"
        sae = Sae.load_from_disk(path, DEVICE)

        def _forward(sae, x):
            encoded = sae.encode(x)
            trained_k = sae.cfg.k
            topk = TopK(trained_k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
            return topk(encoded)

        if "llama" in weight_dir:
            submodule = model.model.layers[layer]
        elif "gpt2" in weight_dir:
            submodule = model.transformer.h[layer]

        submodule.ae = AutoencoderLatents(
            sae, partial(_forward, sae), width=sae.d_in * sae.cfg.expansion_factor
        )

        submodules[layer] = submodule

    with model.edit(" "):
        for _, submodule in submodules.items():
            acts = submodule.output[0]
            submodule.ae(acts, hook=True)

    return submodules
