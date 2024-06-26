from sae_auto_interp.autoencoders.OpenAI.model import Autoencoder
from sae_auto_interp.autoencoders.EleutherAI.model import Sae
import torch

import torch.nn as nn

from sae_auto_interp.autoencoders.OpenAI.model import TopK,ACTIVATIONS_CLASSES

from sae_auto_interp import get_config

class AutoencoderWrapper(nn.Module):
    """Sparse autoencoder from either OAI or EAI
    """
    def __init__(self, autoencoder: nn.Module,type: str) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.n_features = autoencoder.encoder.out_features
        self.type = type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.type == "oai":
            # The OAI autoencoder returns a tuple with the encoded values and statistics, which we don't need
            return self.autoencoder.encode(x)[0]
        elif self.type == "eai":
            # The EAI autoencoder returns the encoded values before topk, and we use OAI TopK class.
            encoded = self.autoencoder.encode(x)
            trained_k = self.autoencoder.cfg.k
            topk = TopK(trained_k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
            return topk(encoded)
        
    
def get_autoencoder(model_name:str,layer: int,device:str,path:str) -> AutoencoderWrapper:
    
    if "gpt2" in model_name:
        autoencoder = load_oai_autoencoder(layer,path)
        autoencoder.to(device)
        wrapped = AutoencoderWrapper(autoencoder,"oai")
    elif "llama" in model_name: 
        autoencoder = load_eai_autoencoder(layer,device,path)
        wrapped = AutoencoderWrapper(autoencoder,"eai")
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    return wrapped


def load_eai_autoencoder(layer: int, device:str, config:dict[str,str],path:str) -> Sae:
    path = f"{path}/Meta-LLama-3-8B/layer_{layer}"
    sae = Sae.load_from_disk(path,device)
    return sae

def load_oai_autoencoder(layer: int, config:dict[str,str],path:str) -> Autoencoder:
    filename = f"{path}/gpt2/resid_post_mlp_autoencoder_{layer}.pt"
    with open(filename, mode="rb") as f:
        state_dict = torch.load(f)
        autoencoder = Autoencoder.from_state_dict(state_dict)
    return autoencoder
    
    


    