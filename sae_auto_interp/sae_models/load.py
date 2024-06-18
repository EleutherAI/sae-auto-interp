from sae_auto_interp.sae_models.open_ai_model import Autoencoder
import torch


def get_autoencoder(model:str,layer: int,device:str) -> Autoencoder:
    if "gpt2" in model:
        autoencoder = load_oai_autoencoder(layer)
        autoencoder.to(device)
    else:
        raise NotImplementedError(f"Model {model} not implemented")


def load_oai_autoencoder(layer: int) -> Autoencoder:
    try:
        # Load the sparse autoencoder
        filename = f"autoencoders/resid_post_mlp_autoencoder_{layer}.pt"
        with open(filename, mode="rb") as f:
            state_dict = torch.load(f)
            autoencoder = Autoencoder.from_state_dict(state_dict)
            return autoencoder
    except FileNotFoundError:
        print(f"Autoencoder for layer {layer} not found")
        return None

