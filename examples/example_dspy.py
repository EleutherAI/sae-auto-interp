#%%
from IPython import get_ipython
from itertools import chain
try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "1")
except AttributeError:
    pass
#%%
from sae_auto_interp.features import (
    FeatureDataset,
    FeatureLoader
)
from sae_auto_interp.features.samplers import sample

from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from functools import partial
from sae_auto_interp.features.constructors import default_constructor
import torch

import dotenv
import os
from sae_auto_interp.clients import DSPy
from dspy import LM

from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.explainers import DefaultExplainer, DSPyExplainer
import asyncio
#%%
dotenv.load_dotenv()
secret_value_0 = os.environ["DSPY_AUTOINTERP_TEST0"]
# llama_8b = dsp.modules.groq_client.GroqLM(secret_value_0,
#                                   "llama3-8b-8192")
llama_8b = LM(
    "llama-3.1-8b-instant",
    # "gemma2-9b-it",
    # "llama-3.3-70b-specdec",
              api_key=secret_value_0, api_base="https://api.groq.com/openai/v1/",
              temperature=0.5
              )
client = DSPy(llama_8b)
print(asyncio.run(client.generate("Hello world!")))
#%%

# start_feature = 13107
start_feature = 49
# start_feature = 0
# n_features = 16383 - 13107
n_features = 16
module = ".model.layers.10"
feature_dict = {f"{module}": torch.arange(start_feature, start_feature + n_features)}
feature_cfg = FeatureConfig.load_json("extra_raw_features/config.json")
feature_cfg.width = 16384
dataset = FeatureDataset(
    raw_dir=f"extra_raw_features",
    cfg=feature_cfg,
    modules=[module],
    features=feature_dict,
)
experiment_cfg = ExperimentConfig(
    train_type="top",
    test_type="quantiles",
    n_examples_train=5,
    n_examples_test=5
)
constructor = partial(
    default_constructor,
    tokens=dataset.tokens,
    n_random=experiment_cfg.n_random,
    ctx_len=experiment_cfg.example_ctx_len,
    max_examples=feature_cfg.max_examples
)
sampler = partial(sample, cfg=experiment_cfg)
loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)
#%%
# explainer_pipe = lambda x: print(x)
def explainer_preprocess(x):
    return x
def explainer_postprocess(x):
    print("before dspy")
    for ex in chain(x.record.train[:2], x.record.train[-2:]):
        print(" > " + repr("".join(
            t if a == 0 else "<<" + t + ">>"
            for t, a in
            zip(dataset.tokenizer.batch_decode(ex.tokens), ex.normalized_activations.tolist())
        )))
    print("after dspy", x)
    return x
explainer_pipe = process_wrapper(
    # DefaultExplainer(
    #     client,
    #     tokenizer=dataset.tokenizer,
    #     threshold=0.5,
    #     activations=True,
    # ),
    DSPyExplainer(
        client.client,
        tokenizer=dataset.tokenizer,
    ),
    preprocess=explainer_preprocess,
    postprocess=explainer_postprocess,
)
pipe = Pipeline(
    loader,
    explainer_pipe
)

asyncio.run(pipe.run(1))