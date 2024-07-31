# %%
from functools import partial

import torch

from sae_auto_interp.config import FeatureConfig
from sae_auto_interp.features import FeatureDataset, pool_max_activation_windows
from sae_auto_interp.utils import load_tokenized_data, load_tokenizer

### Set directories ###

RAW_FEATURES_PATH = "raw_features/gpt2"

cfg= FeatureConfig()

tokenizer = load_tokenizer("gpt2")
tokens = load_tokenized_data(
    cfg.example_ctx_len,
    tokenizer,
    "kh4dien/fineweb-100m-sample",
    "train[:15%]",
)

modules = [f".transformer.h.{i}" for i in range(0, 12, 2)]
features = {m: torch.arange(10) for m in modules}

dataset = FeatureDataset(
    raw_dir=RAW_FEATURES_PATH,
    modules=modules,
    cfg=cfg,
    features=features,
)

generator = partial(
    dataset.load,
    tokens=tokens,
)

data = []

for d in generator():
    data.append(d)

# %%

data
# %%

from sae_auto_interp.utils import display

display(data[0][0], tokenizer)