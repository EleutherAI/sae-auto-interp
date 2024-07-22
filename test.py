
# %%

from sae_auto_interp.load.loader import FeatureDataset
from sae_auto_interp.utils import load_tokenized_data
import torch
from nnsight import LanguageModel

from sae_auto_interp.load.sampling import sample_top_and_quantiles

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
    tokens=tokens,
    sampler=sample_top_and_quantiles
)



# %%

from sae_auto_interp.features.utils import display
records = []
for i in dataset.load():

    records.append(i)

# %%

display(records[0], model.tokenizer)