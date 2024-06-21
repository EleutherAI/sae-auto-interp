from sae_auto_interp.models.open_ai_model import Autoencoder
from sae_auto_interp.models.nora_sae import Sae
import torch


def get_autoencoder(model_name:str,layer: int,device:str):
    if "gpt2" in model_name:
        autoencoder = load_oai_autoencoder(layer)
        autoencoder.to(device)
    elif "llama" in model_name:
        autoencoder = load_nora_autoencoder(layer,device)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    return autoencoder


def load_nora_autoencoder(layer: int,device:str):
    path = f"autoencoders/Meta-LLama-3-8B/layer_{layer}.pt"
    sae = Sae.load_from_disk(path,device)
    return sae

def load_oai_autoencoder(layer: int) -> Autoencoder:
    try:
        # Load the sparse autoencoder
        filename = f"autoencoders/gpt2/resid_post_mlp_autoencoder_{layer}.pt"
        with open(filename, mode="rb") as f:
            state_dict = torch.load(f)
            autoencoder = Autoencoder.from_state_dict(state_dict)
        return autoencoder
    except FileNotFoundError:
        print(f"Autoencoder for layer {layer} not found")
        return None

