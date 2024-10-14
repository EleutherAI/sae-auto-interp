"""
THIS SCRIPT IS DEPRECATED
"""

import asyncio
from functools import partial

import orjson
import torch

from sae_auto_interp.clients import Local,
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import explanation_loader
from sae_auto_interp.features import FeatureDataset, pool_max_activation_windows, sample
from sae_auto_interp.pipeline import  Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import OpenAISimulator
from sae_auto_interp.utils import (
    load_tokenized_data,
    load_tokenizer,
)

### Set directories ###

RAW_FEATURES_PATH = "raw_features/gpt2"
EXPLAINER_OUT_DIR = "results/gpt2_explanations"
SCORER_OUT_DIR = "results/gpt2_simulation/all_at_once"

### Load dataset ###

tokenizer = load_tokenizer('gpt2')
tokens = load_tokenized_data(
    64,
    tokenizer,
    "kh4dien/fineweb-100m-sample",
    "train[:15%]",
)    

modules = [f".transformer.h.{i}" for i in [0, 2, 4, 6, 8, 10]]
features = {
    m : torch.arange(4, 20) for m in modules
}

dataset = FeatureDataset(
    raw_dir=RAW_FEATURES_PATH,
    modules=modules,
    cfg=FeatureConfig(),
    features=features,
)

loader = partial(
    dataset.load,
    constructor=partial(
        pool_max_activation_windows, 
        n_random=5, 
        ctx_len=20, 
        max_examples=5_000
    ),
    sampler=partial(sample, cfg=ExperimentConfig())
)
### Load client ###

client = Local("casperhansen/llama-3-70b-instruct-awq")

### Build Explainer pipe ###

explainer_pipe = partial(explanation_loader, explanation_dir=EXPLAINER_OUT_DIR)

### Build Scorer pipe ###


def scorer_preprocess(result):
    record = result.record

    record.explanation = result.explanation
    record.test = record.test[0][:5]  # use first 5 of top activating quantile
    return record


def scorer_postprocess(result):
    with open(f"{SCORER_OUT_DIR}/{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.score))


scorer_pipe = Pipe(
    process_wrapper(
        OpenAISimulator(client, tokenizer=tokenizer),
        preprocess=scorer_preprocess,
        postprocess=scorer_postprocess,
    )
)

### Build the pipeline ###

pipeline = Pipeline(
    loader,
    explainer_pipe,
    scorer_pipe,
)

asyncio.run(
    pipeline.run()
)
