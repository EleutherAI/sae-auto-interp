from typing import Dict
import torch
from nnsight.editing import Edit

from .oai.model import Autoencoder

class AutoencoderLatents(torch.nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        latents, _  = self.ae.encode(x)
        return latents

def load_autoencoders(model, weight_dir) -> Dict[int, Autoencoder]:
    ae_layers = [0, 2, 4, 6, 8, 10]
    ae_dict = {}
    submodule_dict = {}
    edits = []

    for layer in ae_layers:
        path = f"{weight_dir}/resid_post_mlp_layer{layer}/ae.pt"
        state_dict = torch.load(path)
        ae = Autoencoder.from_state_dict(state_dict=state_dict)
        ae.to("cuda:0")

        submodule = model.transformer.h[layer]
        passthrough_ae = AutoencoderLatents(ae)
        edit = Edit(submodule, "ae", passthrough_ae)

        ae_dict[layer] = ae
        submodule_dict[layer] = submodule
        edits.append(edit)

    return ae_dict, submodule_dict,  edits