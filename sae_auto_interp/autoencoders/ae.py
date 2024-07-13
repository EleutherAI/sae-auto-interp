from typing import Dict, List
import torch

from .OpenAI.model import (
    Autoencoder,
    TopK,
    ACTIVATIONS_CLASSES
)
from .EleutherAI.model import Sae

from .Sam.model import AutoEncoder as SamAE


class AutoencoderLatents(torch.nn.Module):
    """
    Wrapper module to simplify capturing of autoencoder latents.
    """

    def __init__(self, autoencoder: torch.nn.Module,type: str='sam') -> None:
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
        
        elif self.type == "sam":
            latents = self.autoencoder.encode(x)
            return latents
        
    
def load_eai_autoencoders(model,ae_layers: List[int], weight_dir:str):
    
    submodule_dict = {}

    for layer in ae_layers:
        path = f"{weight_dir}/layer_{layer}.pt"
        sae = Sae.load_from_disk(path,"cuda:0")

        submodule = model.model.layers[layer]
        submodule.ae = AutoencoderLatents(sae, "eai")

        submodule_dict[layer] = submodule
    
    with model.edit(" "):
        for _, submodule in submodule_dict.items():
            acts = submodule.output[0]
            submodule.ae(acts, hook=True)

    return submodule_dict	

def load_oai_autoencoders(model,ae_layers: List[int], weight_dir:str):

    submodule_dict = {}
    for layer in ae_layers:
        # Tweaked this to work w how I save my autoencoders locally
        path = f"{weight_dir}/resid_post_mlp_layer{layer}/ae.pt"
        #path = f"{weight_dir}/resid_post_mlp_autoencoder_{layer}.pt"
        state_dict = torch.load(path)
        ae = Autoencoder.from_state_dict(state_dict=state_dict)
        ae.to("cuda:0")

        submodule = model.transformer.h[layer]
        submodule.ae = AutoencoderLatents(ae, "oai")

        submodule_dict[layer] = submodule

    with model.edit(" "):
        for _, submodule in submodule_dict.items():
            acts = submodule.output[0]
            submodule.ae(acts, hook=True)

    return submodule_dict

def load_sam_autoencoders(model, weight_dir, modules=["embed", "mlp", "attention", "resid"]):
    submodules = {}

    dict_id = 10
    dictionary_size = 32768
    DEVICE = "cuda:0"

    layer = 5

    if "embed" in modules:
        submodules[model.gpt_neox.embed_in._module_path] = \
            model.gpt_neox.embed_in
        
        model.gpt_neox.embed_in.ae = AutoencoderLatents(
            SamAE.from_pretrained(
                f'{weight_dir}/embed/{dict_id}_{dictionary_size}/ae.pt',
                device=DEVICE
            )
        )
        
    for i in range(layer + 1):

        if "mlp" in modules:
            submodules[model.gpt_neox.layers[i].mlp._module_path] = \
                model.gpt_neox.layers[i].mlp
            
            model.gpt_neox.layers[i].mlp.ae = AutoencoderLatents(
                SamAE.from_pretrained(
                    f'{weight_dir}/mlp_out_layer{i}/{dict_id}_{dictionary_size}/ae.pt',
                    device=DEVICE
                )
            )

        if "attention" in modules:
            submodules[model.gpt_neox.layers[i].attention._module_path] = \
                model.gpt_neox.layers[i].attention
            
            model.gpt_neox.layers[i].attention.ae = AutoencoderLatents(
                SamAE.from_pretrained(
                    f'{weight_dir}/attn_out_layer{i}/{dict_id}_{dictionary_size}/ae.pt',
                    device=DEVICE
                )
            )

        if "resid" in modules:
            submodules[model.gpt_neox.layers[i]._module_path] = \
                model.gpt_neox.layers[i]
            
            model.gpt_neox.layers[i].ae = AutoencoderLatents(
                SamAE.from_pretrained(
                    f'{weight_dir}/resid_out_layer{i}/{dict_id}_{dictionary_size}/ae.pt',
                    device=DEVICE
                )
            )

        print(f"Loaded autoencoders for layer {i}")

    with model.edit(" "):
        for path, submodule in submodules.items():
            acts = submodule.output
            if "embed" not in path:
                acts = acts[0]
            submodule.ae(acts, hook=True)


    return  submodules


def load_autoencoders(model,ae_layers, weight_dir, **kwargs) -> Dict[int, Autoencoder]:
    
    #TODO: We need to make this a little bit differently in the future
    if "gpt2" in weight_dir:
        submodule_dict = load_oai_autoencoders(model,ae_layers, weight_dir)
       
    if "llama" in weight_dir:
        submodule_dict = load_eai_autoencoders(model,ae_layers, weight_dir)

    if "pythia" in weight_dir:
        submodule_dict = load_sam_autoencoders(model, weight_dir, **kwargs)
    
    return submodule_dict
