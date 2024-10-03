from typing import Callable, Optional, Union, Any, Literal, Dict
import torch
from simple_parsing import Serializable
from functools import partial

class AutoencoderConfig(Serializable):
    model_name_or_path: str 
    autoencoder_type: Literal["SAE", "SAE_LENS", "NEURONS", "CUSTOM"] = "SAE"
    device: Optional[str] = None
    hookpoints: Optional[List[str]] = None
    kwargs: Dict[str, Any] = {}

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
    def from_pretrained(
        cls,
        config: AutoencoderConfig,
        hookpoint: str,
        **kwargs
    ):
        device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        autoencoder_type = config.autoencoder_type
        model_name_or_path = config.model_name_or_path

        if autoencoder_type == "SAE":
            from sae import Sae
            local = kwargs.get("local",None)
            assert local is not None, "local must be specified for SAE"
            if local:
                sae = Sae.load_from_disk(model_name_or_path+"/"+hookpoint, device=device, **kwargs)
            else:
                sae = Sae.load_from_hub(model_name_or_path,hookpoint, device=device, **kwargs)
            forward_function = choose_forward_function(config, sae)
            width = sae.encoder.weight.shape[0]
            
        elif autoencoder_type == "SAE_LENS":
            from sae_lens import SAE
            sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = model_name_or_path, # see other options in sae_lens/pretrained_saes.yaml
            sae_id = hookpoint, 
            device = device
            )
            forward_function = choose_forward_function(config, sae)
            width = sae.d_sae

        elif autoencoder_type == "NEURONS":
            raise NotImplementedError("Neurons autoencoder not implemented yet")

        elif autoencoder_type == "CUSTOM":
            # to use a custom autoencoder, you must make own custom autoencoder class and implement the forward function
            # it should have a specific name that you use here
            custom_name = config.kwargs.get("custom_name", None)
            if custom_name is None:
                raise ValueError("custom_name must be specified for CUSTOM autoencoder")
            if custom_name == "gemmascope":
                from Custom.gemmascope import JumpReLUSAE
                position = config.kwargs.get("position", None)
                assert position is not None, "position must be specified for gemmascope autoencoder"
                sae = JumpReLUSAE.from_pretrained(model_name_or_path,position,device)
                forward_function = choose_forward_function(config, sae)
            if custom_name == "openai":
                raise NotImplementedError("OpenAI autoencoder not implemented yet")
                from Custom.openai import Autoencoder
                path = f"{model_name_or_path}/{hookpoint}.pt"
                state_dict = torch.load(path)
                ae = Autoencoder.from_state_dict(state_dict=state_dict)
        
        else:
            raise ValueError(f"Unsupported autoencoder type: {autoencoder_type}")

        return cls(sae, forward_function, width)
    @classmethod
    def random(cls, config: AutoencoderConfig, hookpoint: str, **kwargs):
        pass

def choose_forward_function(autoencoder_config: AutoencoderConfig, autoencoder: Any):
    if autoencoder_config.autoencoder_type == "SAE":
        from .Custom.openai import ACTIVATIONS_CLASSES, TopK

        def _forward(sae, k,x):
            encoded = sae.pre_acts(x)
            if k is not None:
                trained_k = k
            else:
                trained_k = sae.cfg.k
            topk = TopK(trained_k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
            return topk(encoded)
        k = autoencoder_config.kwargs.get("k", None)
        return partial(_forward, autoencoder, k)

    elif autoencoder_config.autoencoder_type == "SAE_LENS":
        return autoencoder.encode

    elif autoencoder_config.autoencoder_type == "CUSTOM":
        if autoencoder_config.kwargs.get("custom_name", None) == "gemmascope":
            return autoencoder.encode
        else:
            raise ValueError(f"Unsupported custom autoencoder: {autoencoder_config.kwargs.get('custom_name', None)}")

def hook_submodule(autoencoder_config: AutoencoderConfig, submodule: Any, model: Any):
    
    with model.edit("") as edited:
        if "embed" not in submodule.path and "mlp" not in submodule.path:
            acts = submodule.output[0]
        else:
            acts = submodule.output
        submodule.ae(acts, hook=True)
    return submodule,edited


def load_autoencoder_into_model(
    model: Any,
    autoencoder_config: AutoencoderConfig,
    **kwargs
) -> Tuple[List[Any], Any]:
    """
    Load an autoencoder and hook it into the model using nnsight.

    Args:
        model (Any): The main model to hook the autoencoder into.
        hookpoints (list[str]): List of paths to the submodules to hook the autoencoder into.
        autoencoder_config (AutoencoderConfig): Configuration for the autoencoder.

    Returns:
        Tuple[List[Any], Any]: The list of submodules with the autoencoder attached and the edited model.
    """

    submodules = {}
    edited_model = model
    hookpoints = autoencoder_config.hookpoints
    assert hookpoints is not None, "Hookpoints must be specified in autoencoder_config"
    for module_path in hookpoints:
        autoencoder = AutoencoderLatents.from_pretrained(
            autoencoder_config,
            module_path,
        )
        submodule = model.get_submodule(module_path)
        autoencoder.ae = autoencoder
        submodule,edited_model = hook_submodule(autoencoder_config, submodule, edited_model)
        
        submodules[submodule.path] = submodule

    return submodules, edited_model

    
