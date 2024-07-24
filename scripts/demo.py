import asyncio
import orjson

import torch

from sae_auto_interp.explainers import SimpleExplainer
from sae_auto_interp.scorers import RecallScorer
from sae_auto_interp.clients import Local
from sae_auto_interp.utils import load_tokenized_data, load_tokenizer, default_constructor
from sae_auto_interp.features import top_and_quantiles, FeatureLoader, FeatureDataset
from sae_auto_interp.pipeline import Pipe, Pipeline, Actor
from sae_auto_interp.config import FeatureConfig

### Set directories ###

RAW_FEATURES_PATH = "raw_features"
EXPLAINER_OUT_DIR = "results/explanations/simple"
SCORER_OUT_DIR = "results/scores"
SCORER_OUT_DIR_B = "results/scores_b"

### Load dataset ###

tokenizer = load_tokenizer('gpt2')
tokens = load_tokenized_data(tokenizer)

modules = [".transformer.h.0", ".transformer.h.2"]
features = {
    m : torch.arange(10) for m in modules
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
    constructor=default_constructor,
    sampler=top_and_quantiles
)

### Load client ###

client = Local("meta-llama/Meta-Llama-3-8B-Instruct")

### Build Explainer pipe ###

def explainer_postprocess(result):
    result = result.result()
    with open(f"{EXPLAINER_OUT_DIR}/{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.explanation))

explainer_pipe = Pipe(
    Actor(
        SimpleExplainer(client, tokenizer=tokenizer),
        postprocess=explainer_postprocess
    )
)

### Build Scorer pipe ###

def scorer_preprocess(result):
    record = result.record
    record.explanation = result.explanation
    return record

def scorer_postprocess(result):
    result = result.result()
    with open(f"{SCORER_OUT_DIR}/{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.score))

def scorer_postprocess_b(result):
    result = result.result()
    with open(f"{SCORER_OUT_DIR_B}/{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.score))

scorer_pipe = Pipe(
    Actor(
        RecallScorer(client, tokenizer=tokenizer),
        preprocess=scorer_preprocess,
        postprocess=scorer_postprocess
    ),
    Actor(
        RecallScorer(client, tokenizer=tokenizer),
        preprocess=scorer_preprocess,
        postprocess=scorer_postprocess_b
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

