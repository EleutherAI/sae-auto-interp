from typing import Dict, List
import torch

from .OpenAI.model import (
    Autoencoder,
    TopK,
    ACTIVATIONS_CLASSES
)
from .EleutherAI.model import Sae
class AutoencoderLatents(torch.nn.Module):
    def __init__(self, autoencoder: torch.nn.Module,type: str) -> None:
            super().__init__()
            self.autoencoder = autoencoder
            self.n_features = autoencoder.encoder.out_features
            self.type = type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.type == "oai":
            # The OAI autoencoder returns a tuple with the encoded values and statistics, which we don't need
            latents,_ = self.autoencoder.encode(x)
            return latents
        elif self.type == "eai":
            # The EAI autoencoder returns the encoded values before topk, and we use OAI TopK class.
            encoded = self.autoencoder.encode(x)
            trained_k = self.autoencoder.cfg.k
            topk = TopK(trained_k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
            return topk(encoded)
        
    
def load_eai_autoencoders(model,ae_layers: List[int], weight_dir:str):
    
    ae_dict = {}
    submodule_dict = {}

    for layer in ae_layers:
        path = f"{weight_dir}/layer_{layer}.pt"
        sae = Sae.load_from_disk(path,"cuda:0")

        submodule = model.model.layers[layer]
        passthrough_ae = AutoencoderLatents(sae, "eai")
        submodule.ae = passthrough_ae

        ae_dict[layer] = sae
        submodule_dict[layer] = submodule

    with model.alter(" "):
        for layer_idx, _ in ae_dict.items():
            layer = model.model.layers[layer_idx]
            acts = layer.output[0]
            layer.ae(acts, hook=True)

    return ae_dict, submodule_dict	

def load_oai_autoencoders(model,ae_layers: List[int], weight_dir:str):
    ae_dict = {}
    submodule_dict = {}
    for layer in ae_layers:
        # Tweaked this to work w how I save my autoencoders locally
        path = f"{weight_dir}/resid_post_mlp_layer{layer}/ae.pt"
        # path = f"{weight_dir}/resid_post_mlp_{layer}.pt"
        state_dict = torch.load(path)
        ae = Autoencoder.from_state_dict(state_dict=state_dict)
        ae.to("cuda:0")

        submodule = model.transformer.h[layer]
        passthrough_ae = AutoencoderLatents(ae, "oai")
        submodule.ae = passthrough_ae

        ae_dict[layer] = ae
        submodule_dict[layer] = submodule

    with model.alter(" "):
        for layer_idx, _ in ae_dict.items():
            layer = model.transformer.h[layer_idx]
            acts = layer.output[0]
            layer.ae(acts, hook=True)

    return ae_dict, submodule_dict


def load_autoencoders(model,ae_layers, weight_dir) -> Dict[int, Autoencoder]:
    
    #TODO: We need to make this a little bit differently in the future
    if "gpt2" in weight_dir:
        ae_dict, submodule_dict = load_oai_autoencoders(model,ae_layers, weight_dir)
       
    if "llama" in weight_dir:
        ae_dict, submodule_dict = load_eai_autoencoders(model,ae_layers, weight_dir)
    
    return ae_dict, submodule_dict
