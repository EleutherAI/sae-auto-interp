from typing import Dict, List
import torch
from nnsight.editing import Edit

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
    edits = []
    for layer in ae_layers:
        path = f"{weight_dir}/layer_{layer}"
        sae = Sae.load_from_disk(path,"cuda:0")

        submodule = model.model.layers[layer]
        passthrough_ae = AutoencoderLatents(sae, "eai")
        edit = Edit(submodule, "ae", passthrough_ae)

        ae_dict[layer] = sae
        submodule_dict[layer] = submodule
        edits.append(edit)

    return ae_dict, submodule_dict, edits	

def load_oai_autoencoders(model,ae_layers: List[int], weight_dir:str):
    ae_dict = {}
    submodule_dict = {}
    edits = []
    for layer in ae_layers:
        path = f"{weight_dir}/resid_post_mlp_autoencoder_{layer}.pt"
        state_dict = torch.load(path)
        ae = Autoencoder.from_state_dict(state_dict=state_dict)
        ae.to("cuda:0")

        submodule = model.transformer.h[layer]
        passthrough_ae = AutoencoderLatents(ae, "oai")
        edit = Edit(submodule, "ae", passthrough_ae)

        ae_dict[layer] = ae
        submodule_dict[layer] = submodule
        edits.append(edit)

    return ae_dict, submodule_dict, edits


def load_autoencoders(model,ae_layers, weight_dir) -> Dict[int, Autoencoder]:
    
    #TODO: We need to make this a little bit differently in the future
    if "gpt2" in weight_dir:
        ae_dict, submodule_dict, edits = load_oai_autoencoders(model,ae_layers, weight_dir)
       
    if "Llama" in weight_dir:
        ae_dict, submodule_dict, edits = load_eai_autoencoders(model,ae_layers, weight_dir)
    

    return ae_dict, submodule_dict,  edits
