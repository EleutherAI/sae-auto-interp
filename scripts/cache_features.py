
import torch
from nnsight import LanguageModel
#from nnsight.editing import Edit
from sae_auto_interp.features.cache import FeatureCache
from sae_auto_interp.autoencoders.model import get_autoencoder
from transformer_lens import HookedTransformer
#model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
import json
from argparse import ArgumentParser

class AutoencoderLatents(torch.nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        latents, _  = self.ae.encode(x)
        return latents

# def load_autoencoders() -> Dict[int, Autoencoder]:
#     ae_layers = [0, 2, 4, 6, 8, 10]
#     ae_dict = {}
#     edits = []

#     for layer in ae_layers:
#         path = f"./autointerp/autoencoders/oai/gpt2/resid_post_mlp_layer{layer}/ae.pt"
#         state_dict = torch.load(path)
#         ae = Autoencoder.from_state_dict(state_dict=state_dict)
#         ae.to("cuda:0")

#         submodule = model.transformer.h[layer]
#         passthrough_ae = AutoencoderLatents(ae)
#         edit = Edit(submodule, "ae", passthrough_ae)

#         ae_dict[layer] = submodule
#         edits.append(edit)

#     return ae_dict, edits



#ae_dict, edits = load_autoencoders()

# with model.alter(" ", edits=edits):
#     for layer_idx, _ in ae_dict.items():
#         layer = model.transformer.h[layer_idx]
#         acts = layer.output[0]
#         layer.ae(acts, hook=True)


#TODO: This is a temporary script to cache the features. Should probably think what the final shape will be 
if __name__ == "__main__":
    parser = ArgumentParser()
    #TODO: Allow for a list of layers
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer to collect features from, if -1 will collect from all layers",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2","meta-llama/Meta-Llama-3-8B"]
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    #Not used at the moment
    parser.add_argument(
        "--number_features",
        type=int,
        default=-1,
        help="Number of features to collect, if -1 will collect all features",
    )
    #Not used at the moment
    parser.add_argument(
        "--feature_prefix",
        type=str,
        default="",
        help="Prefix to add to the file name",
    )
    parser.add_argument(
        "--path_autoencoder",
        type=str,
        default="/mnt/ssd-1/gpaulo/SAE-Zoology/saved_autoencoders",
        help="Path to the autoencoder files",
    )

    args = parser.parse_args()
    model_name = args.model
    device = args.device
    layer = args.layer
    number_features = args.number_features
    config = args.dataset_configuration
    prefix = args.feature_prefix
    if prefix != "":
        prefix = prefix + "_"
    autoencoder_path= args.path_autoencoder


    model = HookedTransformer.from_pretrained(
            model_name, center_writing_weights=False, device=device, dtype="bfloat16"
        )
    tokenizer = model.tokenizer

    print("Loading the autoencoders")
    if layer == -1:
        layers = list(range(12))
    else:
        layers = [layer]
    #TODO: We should probably load the autoencoders in a different way. Open to ideas.
    ae_dict = {
        layer: get_autoencoder(model_name, layer, device,autoencoder_path) for layer in layers
    }


    cache = FeatureCache(model, ae_dict)
    cache.run(use_transformer_lens=True)

    with open("../samples.json", "r") as f:
        features = json.load(f)

    cache.save_some_features(features)

# for layer_index in [0,2,4,6,8,10]:
#     feature_activations = cache.buffer.feature_activations[layer_index]
#     feature_indexs = cache.buffer.feature_locations[layer_index]
#     torch.save(feature_activations, f"./cache/layer{layer_index}_activations.pt")
#     torch.save(feature_indexs, f"./cache/layer{layer_index}_indexes.pt")