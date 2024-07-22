# %%
import asyncio
import torch
from sae_auto_interp.explainers import SimpleExplainer
# from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.clients import Local
from sae_auto_interp.utils import load_tokenized_data, load_tokenizer, default_constructor
from sae_auto_interp.features import top_and_quantiles, FeatureLoader, FeatureDataset
from sae_auto_interp.pipeline import Pipe, Pipeline

tokenizer = load_tokenizer('gpt2')
tokens = load_tokenized_data(tokenizer)

raw_features_path = "raw_features"
explainer_out_dir = "results/explanations/simple"

modules = [".transformer.h.0", ".transformer.h.2"]

features = {
    m : torch.arange(10) for m in modules
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

def explainer_postprocess(result):
    result = result.result()
    with open(f"{explainer_out_dir}/{result.record.feature}.txt", "w") as f:
        f.write(result.explanation)

# def scorer_preprocess(record):
#     return ScorerInput(
#         record=record,
#         test_examples=sum(record.test, []),
#         explanation=record.explanation,
#         random_examples=record.random_examples,
#     )

client = Local("meta-llama/Meta-Llama-3-8B-Instruct")

explainer_pipe = Pipe(
    [SimpleExplainer(client, tokenizer=tokenizer)],
    postprocess=explainer_postprocess
)

# scorer_pipe = Pipe(
#     scorer_preprocess,
#     FuzzingScorer(client, tokenizer=tokenizer)
# )

pipeline = Pipeline(
    loader.load,
    explainer_pipe,
    # scorer_pipe,
)

asyncio.run(
    pipeline.run()
)

