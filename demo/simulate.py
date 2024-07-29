import asyncio
import orjson

import torch

from defaults import default_constructor
from sae_auto_interp.explainers import explanation_loader
from sae_auto_interp.scorers import OpenAISimulator
from sae_auto_interp.clients import Outlines, Local
from sae_auto_interp.utils import load_tokenized_data, load_tokenizer
from sae_auto_interp.features import top_and_quantiles, FeatureLoader, FeatureDataset
from sae_auto_interp.pipeline import Pipe, process_wrapper, Pipeline
from functools import partial
from sae_auto_interp.config import FeatureConfig

### Set directories ###

RAW_FEATURES_PATH = "/mnt/ssd-1/gpaulo/SAE-Zoology/raw_features_128k"
EXPLAINER_OUT_DIR = "/mnt/ssd-1/gpaulo/SAE-Zoology/results/gpt2_explanations"
SCORER_OUT_DIR = "/mnt/ssd-1/gpaulo/SAE-Zoology/results/gpt2_simulation"

### Load dataset ###

tokenizer = load_tokenizer('gpt2')
tokens = load_tokenized_data(
    64,
    tokenizer,
    "kh4dien/fineweb-100m-sample",
    "train",
)    

modules = [".transformer.h.0"]
features = {
    m : torch.arange(1) for m in modules
}

dataset = FeatureDataset(
    raw_dir=RAW_FEATURES_PATH,
    modules = modules,
    cfg=FeatureConfig(),
    features=features,
)

loader = FeatureLoader(
    tokens=tokens,
    dataset=dataset,
    constructor=partial(
        default_constructor, 
        n_random=5, 
        ctx_len=20, 
        max_examples=5_000
    ),
    sampler=top_and_quantiles
)

### Load client ###

client = Outlines("casperhansen/llama-3-70b-instruct-awq")

### Build Explainer pipe ###

explainer_pipe = partial(explanation_loader, explanation_dir=EXPLAINER_OUT_DIR)

### Build Scorer pipe ###

def scorer_preprocess(result):
    record = result.record

    record.explanation = result.explanation
    record.test = record.test[0][:5]
    return record

def scorer_postprocess(result):
    result = result.result()
    with open(f"{SCORER_OUT_DIR}/{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.score))

scorer_pipe = Pipe(
    process_wrapper(
        OpenAISimulator(client, tokenizer=tokenizer),
        preprocess=scorer_preprocess,
        postprocess=scorer_postprocess
    )
)

### Build the pipeline ###

pipeline = Pipeline(
    loader.load,
    explainer_pipe,
    scorer_pipe,
)

asyncio.run(
    pipeline.run()
)

