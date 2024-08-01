# %%

import pickle as pkl
from collections import defaultdict
from functools import partial

from tqdm import tqdm

from sae_auto_interp.config import FeatureConfig
from sae_auto_interp.features import (
    FeatureDataset,
    FeatureLoader,
    pool_max_activation_windows,
)
from sae_auto_interp.features.stats import unigram
from sae_auto_interp.utils import load_tokenized_data, load_tokenizer

RAW_FEATURES_PATH = "raw_features/gpt2"


tokenizer = load_tokenizer("gpt2")
tokens = load_tokenized_data(
    64,
    tokenizer,
    "kh4dien/fineweb-100m-sample",
    "train[:15%]",
)

modules = [f".transformer.h.{i}" for i in range(0, 12, 2)]

cfg = FeatureConfig(
    width=131_072,
    min_examples=100,
    max_examples=5000,
    ctx_len=64,
    n_splits=2,
    n_train=4,
    n_test=5,
    n_quantiles=10,
)

dataset = FeatureDataset(
    raw_dir=RAW_FEATURES_PATH,
    modules=modules,
    cfg=cfg,
)

loader = FeatureLoader(
    tokens=tokens,
    dataset=dataset,
    constructor=partial(pool_max_activation_windows, ctx_len=20, max_examples=100_000),
)

sparse = defaultdict(lambda: defaultdict(dict))

for batch in loader.load():
    for record in tqdm(batch):
        layer = record.feature.module_name
        feature = record.feature.feature_index

        unique, nonzero = unigram(record, 20, 0.8, negative_shift=1)

        if nonzero < 2 and unique != -1:
            sparse[layer][feature] = unique


with open("sparse_08.pkl", "wb") as f:
    pkl.dump(dict(sparse), f)
