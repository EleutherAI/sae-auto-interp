# %%

import torch
from tqdm import tqdm
from collections import defaultdict
from functools import partial

from sae_auto_interp.utils import load_tokenized_data, load_tokenizer, default_constructor
from sae_auto_interp.features import FeatureLoader, FeatureDataset
from sae_auto_interp.config import FeatureConfig
from sae_auto_interp.features.stats import unigram


RAW_FEATURES_PATH = "raw_features/gpt2"


tokenizer = load_tokenizer('gpt2')
tokens = load_tokenized_data(tokenizer)

modules = [f".transformer.h.{i}" for i in range(0,12,2)]

features = {
    m : torch.arange(10) for m in modules
}

cfg = FeatureConfig()

dataset = FeatureDataset(
    raw_dir=RAW_FEATURES_PATH,
    modules = modules,
    cfg=cfg,
)

loader = FeatureLoader(
    tokens=tokens,
    dataset=dataset,
    constructor=partial(default_constructor, n_random=0, ctx_len=20, max_examples=5_000),
)

sparse = defaultdict(lambda : defaultdict(dict))

for batch in loader.load():

    for k in tqdm(range(150, 3150, 500)):

        for record in batch: 

            layer = record.feature.module_name 
            feature = record.feature.feature_index

            n_unique, _ = unigram(record, k)
            sparse[k][layer][feature] = n_unique

