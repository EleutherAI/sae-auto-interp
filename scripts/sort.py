# %%
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.utils import load_tokenized_data
from nnsight import LanguageModel
from sae_auto_interp.utils import load_tokenized_data, get_samples

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

tokens = load_tokenized_data(
    model.tokenizer
)

samples = get_samples(features_per_layer=20)

# %%
records = FeatureRecord.from_tensor(
    tokens,
    model.tokenizer,
    0,
    raw_dir="/share/u/caden/sae-auto-interp/raw_features/",
    selected_features=samples[0],
    processed_dir="/share/u/caden/sae-auto-interp/processed_features",
    max_examples=1000
)

# %%

from sae_auto_interp.features.stats import CombinedStat, Activations, Skew, Neighbors, Kurtosis
from sae_auto_interp.autoencoders.ae import load_autoencoders

ae_dict, _, _ = load_autoencoders(model, "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2")

stat = CombinedStat(
    skew=Skew(),
    kurtosis=Kurtosis(),
    # neighbors=Neighbors(),
    acts=Activations(
        lemmatize=True
    )
)


stat.refresh(W_U=model.lm_head.weight, W_dec=ae_dict[0].decoder.weight)
stat.compute(records)