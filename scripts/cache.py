from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features.cache import FeatureCache
from sae_auto_interp.sample import get_samples
from nnsight import LanguageModel

# Load model and autoencoders
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

ae_dict, submodule_dict, edits = load_autoencoders(
    model, 
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2"
)

# Set a default alteration on the model
with model.alter(" ", edits=edits):
    for layer_idx, _ in ae_dict.items():
        layer = model.transformer.h[layer_idx]
        acts = layer.output[0]
        layer.ae(acts, hook=True)


# Get and sort samples
samples = get_samples()
samples = {
    int(layer) : features[:10] 
    for layer, features in samples.items() 
    if int(layer) in ae_dict.keys()
}

# Cache and save features
# NOTE: I have the cache only process 10k tokens for testing
cache = FeatureCache(model, ae_dict)
cache.run()
cache.save_some_features(samples, save_dir="/share/u/caden/sae-auto-interp/saved_records")