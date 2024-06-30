from nnsight import LanguageModel 
from tqdm import tqdm

from sae_auto_interp.utils import load_tokenized_data, get_samples
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import CombinedStat, Logits, feature_loader, Feature

# Load model and autoencoders
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
ae_dict, submodule_dict, edits = load_autoencoders(
    model, 
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2"
)

# Load tokenized data
tokens = load_tokenized_data(model.tokenizer)

# Load features I want to explain
samples = get_samples(features_per_layer=20)
samples = {layer : samples[layer] for layer in samples if int(layer) in [0,2,4,6,8,10]}
features = Feature.from_dict(samples)

raw_features_path = "/share/u/caden/sae-auto-interp/raw_features"
processed_features_path = "/share/u/caden/sae-auto-interp/processed_features"

# You can add any object that inherits from Stat
# to combined stats. This info is added to the record
stats = CombinedStat(
    logits = Logits(
        model=model,
        top_k_logits=10
    )
)    

explainer_inputs = []
for ae, records in feature_loader(
    tokens, 
    features,
    model,
    ae_dict,
    raw_features_path,
    pipe=True
):
    # Refresh updates a memory intensive caches for stuff like
    # umap locations or logit matrices
    stats.refresh(W_dec=ae.decoder.weight)

    # Compute updates records with stat information
    stats.compute(records)

    # Save the processed information to the processed feature dir
    for record in records:
        record.save(processed_features_path)