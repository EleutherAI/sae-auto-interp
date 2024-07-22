# %%
import asyncio
import torch
from sae_auto_interp.explainers import SimpleExplainer, ExplainerInput
from sae_auto_interp.clients import get_client, execute_model
from sae_auto_interp.utils import load_tokenized_data, load_tokenizer, default_constructor
from sae_auto_interp.features import top_and_quantiles, FeatureLoader, FeatureDataset
from sae_auto_interp.pipeline import Pipe, Pipeline

tokenizer = load_tokenizer('gpt2')
tokens = load_tokenized_data(tokenizer)

raw_features_path = "raw_features"
explainer_out_dir = "results/explanations/simple"

modules = [".transformer.h.0", ".transformer.h.2"]

features = {
    m : torch.arange(100) for m in modules
}

dataset = FeatureDataset(
    raw_dir="raw_features",
    modules = modules,
    features=features,
)

loader = FeatureLoader(
    tokens=tokens,
    dataset=dataset,
    constructor=default_constructor,
    sampler=top_and_quantiles
)

def preprocess(record):
    return ExplainerInput(
        train_examples=record.train,
        record=record,
    )

client = get_client("local", "meta-llama/Meta-Llama-3-8B-Instruct")

pipe = Pipe(
    preprocess,
    SimpleExplainer(client, tokenizer=tokenizer)
)

pipeline = Pipeline(
    generator=loader.load,
    pipes=[pipe]
)

asyncio.run(
    pipeline.run()
)

