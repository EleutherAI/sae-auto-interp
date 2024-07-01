# %%
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.utils import load_tokenized_data
from nnsight import LanguageModel
from sae_auto_interp.utils import load_tokenized_data, get_samples

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

tokens = load_tokenized_data(
    model.tokenizer
)

samples = get_samples(features_per_layer=100)

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

for record in records:
    if record.feature.feature_index == 1547:
        record.display(n_examples=50)
