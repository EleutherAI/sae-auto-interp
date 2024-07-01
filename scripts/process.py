from nnsight import LanguageModel 
from tqdm import tqdm

from sae_auto_interp.utils import load_tokenized_data, get_samples
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import CombinedStat, Logits, Feature, FeatureRecord

# Load model and autoencoders
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
ae_dict, submodule_dict, edits = load_autoencoders(
    model, 
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2"
)

# Load tokenized data
tokens = load_tokenized_data(model.tokenizer)

# Load features I want to explain
samples = get_samples(features_per_layer=100)
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

for layer, ae in ae_dict.items():

    selected_features = features[layer]

    records = FeatureRecord.from_tensor(
        tokens,
        model.tokenizer,
        layer,
        f"/share/u/caden/sae-auto-interp/raw_features/layer{layer}_locations.pt",
        f"/share/u/caden/sae-auto-interp/raw_features/layer{layer}_activations.pt",
        max_examples=2000
    )
    # Refresh updates a memory intensive caches for stuff like
    # umap locations or logit matrices
    stats.refresh(W_dec=ae.decoder.weight)

    # Compute updates records with stat information
    stats.compute(records)

    # Save the processed information to the processed feature dir
    for record in records:
        record.save(processed_features_path)