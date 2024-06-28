
import torch
from nnsight import LanguageModel
#from nnsight.editing import Edit
from sae_auto_interp.features.cache import FeatureCache
from sae_auto_interp.autoencoders.model import get_autoencoder
from transformer_lens import HookedTransformer
#model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

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


model_name = "gpt2"
device = "cuda:1"


model = HookedTransformer.from_pretrained(
        model_name, center_writing_weights=False, device=device, dtype="bfloat16"
    )
tokenizer = model.tokenizer

print("Loading the autoencoders")
autoencoder_path = "/mnt/ssd-1/gpaulo/SAE-Zoology/saved_autoencoders"
layers = list(range(12))
ae_dict = {
    layer: get_autoencoder(model_name, layer, device,autoencoder_path) for layer in layers
}


cache = FeatureCache(model, ae_dict)
cache.run(use_transformer_lens=True)

# with open("./autointerp/scripts/samples.json", "r") as f:
#     features = json.load(f)

# cache.save_some_features(features)

for layer_index in [0,2,4,6,8,10]:
    feature_activations = cache.buffer.feature_activations[layer_index]
    feature_indexs = cache.buffer.feature_locations[layer_index]
    torch.save(feature_activations, f"./cache/layer{layer_index}_activations.pt")
    torch.save(feature_indexs, f"./cache/layer{layer_index}_indexs.pt")