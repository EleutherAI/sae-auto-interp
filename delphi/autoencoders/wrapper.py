from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Literal, Optional

import torch
from simple_parsing import Serializable

from delphi.autoencoders.load_sparsify import resolve_path


@dataclass
class AutoencoderConfig(Serializable):
    model_name_or_path: str = "model"
    autoencoder_type: Literal["SAE", "SAE_LENS", "NEURONS", "CUSTOM"] = "SAE"
    device: Optional[str] = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class AutoencoderLatents(torch.nn.Module):
    """
    Unified wrapper for different types of autoencoders, compatible with nnsight.
    """

    def __init__(
        self,
        autoencoder: Any,
        forward_function: Callable,
        width: int,
    ) -> None:
        super().__init__()
        self.ae = autoencoder
        self._forward = forward_function
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

    @classmethod
    def from_pretrained(cls, config: AutoencoderConfig, hookpoint: str, **kwargs):
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder_type = config.autoencoder_type
        model_name_or_path = config.model_name_or_path
        if autoencoder_type == "SAE":
            from sparsify import Sae

            local = kwargs.get("local", None)
            assert local is not None, "local must be specified for SAE"
            if local:
                sae = Sae.load_from_disk(
                    model_name_or_path + "/" + hookpoint, device=device, **kwargs
                )
            else:
                sae = Sae.load_from_hub(
                    model_name_or_path, hookpoint, device=device, **kwargs
                )
            forward_function = choose_forward_function(config, sae)
            width = sae.encoder.weight.shape[0]

        elif autoencoder_type == "SAE_LENS":
            from sae_lens import SAE

            sae, cfg_dict, sparsity = SAE.from_pretrained(
                # see other options in sae_lens/pretrained_saes.yaml
                release=model_name_or_path,
                sae_id=hookpoint,
                device=device,
            )
            forward_function = choose_forward_function(config, sae)
            width = sae.d_sae

        elif autoencoder_type == "NEURONS":
            raise NotImplementedError("Neurons autoencoder not implemented yet")

        elif autoencoder_type == "CUSTOM":
            # to use a custom autoencoder, you must make own custom autoencoder class
            # and implement the forward function
            # it should have a specific name that you use here
            custom_name = config.kwargs.get("custom_name", None)
            if custom_name is None:
                raise ValueError("custom_name must be specified for CUSTOM autoencoder")
            if custom_name == "gemmascope":
                from .Custom.gemmascope import JumpReluSae

                sae = JumpReluSae.from_pretrained(model_name_or_path, hookpoint, device)
                forward_function = choose_forward_function(config, sae)
                width = sae.W_enc.data.shape[1]

        else:
            raise ValueError(f"Unsupported autoencoder type: {autoencoder_type}")

        return cls(sae, forward_function, width, hookpoint)

    @classmethod
    def random(cls, config: AutoencoderConfig, hookpoint: str, **kwargs):
        pass


def choose_forward_function(cfg: AutoencoderConfig, autoencoder: Any):
    if cfg.autoencoder_type == "SAE":
        from .eleuther import sae_dense_latents

        return partial(sae_dense_latents, autoencoder)

    elif cfg.autoencoder_type == "SAE_LENS":
        return autoencoder.encode

    elif cfg.autoencoder_type == "CUSTOM":
        if cfg.kwargs.get("custom_name", None) == "gemmascope":
            return autoencoder.encode
        else:
            raise ValueError(
                f"Unsupported custom autoencoder {cfg.kwargs.get('custom_name', None)}"
            )


def load_autoencoder_into_model(
    model: Any, autoencoder_config: AutoencoderConfig, hookpoints: list[str], **kwargs
) -> dict[str, Any]:
    """
    Load autoencoders and add them to a dict keyed by hookpoint.

    Args:
        model (Any): The main model to hook the autoencoder into.
        autoencoder_config (AutoencoderConfig): Configuration for the autoencoder.

    Returns:
        dict[str, Any]: A dict of hookpoints with the autoencoders attached.
    """

    submodules = {}
    assert hookpoints is not None, "Hookpoints must be specified in autoencoder_config"
    for hookpoint in hookpoints:
        path_segments = resolve_path(model, hookpoint.split("."))
        if path_segments is None:
            raise ValueError(f"Could not find valid path for hookpoint: {hookpoint}")
        resolved_hookpoint = ".".join(path_segments)

        autoencoder = AutoencoderLatents.from_pretrained(
            autoencoder_config,
            hookpoint,
        )

        submodules[resolved_hookpoint] = autoencoder.forward

    return submodules
