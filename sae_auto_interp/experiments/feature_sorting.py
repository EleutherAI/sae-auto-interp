from nnsight import LanguageModel 
from tqdm import tqdm
from sae_auto_interp.utils import load_tokenized_data, get_samples
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import (
    CombinedStat, Feature, FeatureRecord
)
from sae_auto_interp.features.stats import Activation, Logits

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
ae_dict, submodule_dict, edits = load_autoencoders(
    model, 
    list(range(0,12,2)),
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2"
)

tokens = load_tokenized_data(model.tokenizer)
samples = get_samples()

raw_features_path = "/share/u/caden/sae-auto-interp/raw_features"
processed_features_path = "/share/u/caden/sae-auto-interp/feature_statistics"

stat = CombinedStat(
    activation = Activation(
        k=1000,
        get_kurtosis=True,
        get_skew=True,
        get_similarity=True,
    ),
    logits = Logits(
        model, 
        get_skew=True,
        get_kurtosis=True,
        get_entropy=True,
        get_perplexity=True
    )
)


for layer, ae in ae_dict.items():
    selected_features = samples[layer]

    records = FeatureRecord.from_tensor(
        tokens,
        layer_index=layer,
        tokenizer=model.tokenizer,
        raw_dir=raw_features_path,
        selected_features=selected_features,
        min_examples=300,
        max_examples=1000,
    )

    stat.refresh(
        W_dec = ae.decoder.weight
    )

    stat.compute(records)
    
    for record in records:
        record.save(processed_features_path)
