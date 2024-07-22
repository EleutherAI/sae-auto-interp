
# %%

from sae_auto_interp.features.loader import FeatureLoader, FeatureDataset
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features.constructors import pool_max_activation_windows, random_activation_windows
from sae_auto_interp.features.samplers import top_and_quantiles
import torch
from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map="auto")

tokens = load_tokenized_data(model.tokenizer)

modules = [".transformer.h.0", ".transformer.h.2"]

features = {
    m : torch.arange(100) for m in modules
}

dataset = FeatureDataset(
    raw_dir="raw_features",
    modules = modules,
    features=features,
)

def constructor(record, tokens, locations, activations):

    pool_max_activation_windows(
        record,
        tokens=tokens,
        locations=locations,
        activations=activations,
        k=200
    )

    random_activation_windows(
        record,
        tokens=tokens,
        locations=locations,
    )


loader = FeatureLoader(
    tokens=tokens,
    dataset=dataset,
    constructor=constructor,
    sampler=top_and_quantiles
)

# %%

records = loader.load_all()

# %%

len(records)