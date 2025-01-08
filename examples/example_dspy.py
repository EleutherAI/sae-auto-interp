#%%
import sys
import os
if os.getcwd().endswith("-auto-interp"):
    os.chdir("examples")

if ".." not in sys.path:
    sys.path.append("..")
from itertools import chain

from IPython import get_ipython

try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "1")
except AttributeError:
    pass
#%%
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import dotenv
import torch
from dspy import LM

from sae_auto_interp.clients import DSPy
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer, DSPyExplainer
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.scorers import DetectionScorer, FuzzingScorer, DSPyClassifier

#%%
USE_OPENROUTER = True
USE_BIG_LLAMA = False
LOAD_PYTHIA = False
#%%
def top_level_await(fn):
    # https://stackoverflow.com/a/61331974
    try:
        asyncio.get_running_loop()
        with ThreadPoolExecutor(1) as pool:
            return pool.submit(lambda: asyncio.run(fn)).result()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        return asyncio.run(fn)


dotenv.load_dotenv()
if USE_OPENROUTER:
    dspy_lm = LM(
        "openrouter/" + (
            "meta-llama/llama-3.3-70b-instruct" if USE_BIG_LLAMA
            else "meta-llama/llama-3.1-8b-instruct"
        ),
        api_key=os.environ["OPENROUTER_API_KEY"]
    )
else:
    dspy_lm = LM(
        "llama-3.3-70b-specdec" if USE_BIG_LLAMA else "llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        api_base="https://api.groq.com/openai/v1/",
    )
client = DSPy(dspy_lm)
# print(asyncio.run(client.generate("Hello world!")))
# print(top_level_await(client.generate("Hello world!")))
#%%

if LOAD_PYTHIA:
    # cache_dir = "/mnt/ssd-1/gpaulo/SAE-Zoology/extras/transcoders" \
    #             "/raw_features/pythia_pile/SAE-2-seed-4k"
    cache_dir = "/mnt/ssd-1/gpaulo/SAE-Zoology/extras/transcoders" \
                "/raw_features/pythia_pile/SkipTranscoder"
    module = ".gpt_neox.layers.6.mlp"
else:
    cache_dir = "../extra_raw_features"
    module = ".model.layers.10"

if LOAD_PYTHIA:
    start_feature = 0
    n_features = 16
else:
    # start_feature = 13107
    start_feature = 0
    n_features = 16
feature_dict = {f"{module}": torch.arange(start_feature, start_feature + n_features)}
feature_cfg = FeatureConfig.load_json(f"{cache_dir}/{module}/config.json")
if not LOAD_PYTHIA:
    feature_cfg.width = 16384
dataset = FeatureDataset(
    raw_dir=cache_dir,
    cfg=feature_cfg,
    modules=[module],
    features=feature_dict,
)
experiment_cfg = ExperimentConfig(
    train_type="top",
    test_type="quantiles",
    n_examples_train=20,
    n_examples_test=10,
    n_quantiles=5,
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
def visualize_record(record):
    print(record)
    for ex in chain(record.train[:2], record.train[-2:]):
        print(
            " > "
            + repr(
                "".join(
                    t if a == 0 else "<<" + t + ">>"
                    for t, a in zip(
                        dataset.tokenizer.batch_decode(ex.tokens),
                        ex.normalized_activations.tolist(),
                    )
                )
            )
        )



async def visualize_loader(loader):
    async for record in loader:
        visualize_record(record)

# top_level_await(visualize_loader(loader))
#%%
# explainer_pipe = lambda x: print(x)
def explainer_preprocess(x):
    return x
def explainer_postprocess(x):
    print("Before dspy:")
    visualize_record(x.record)
    print("After dspy:", x)
    return x
default_explainer = DefaultExplainer(
    client,
    tokenizer=dataset.tokenizer,
    threshold=0.5,
    activations=True,
)
dspy_explainer = DSPyExplainer(
    client.client,
    tokenizer=dataset.tokenizer,
    verbose=True,
)

def scorer_preprocess(result):
    record = result.record
    record.explanation = result.explanation
    record.extra_examples = record.random_examples
    return record
def scorer_postprocess(x):
    corrects = [c.correct for c in x.score]
    print("Accuracy:", sum(map(int, corrects)) / len(corrects))
    trues = [c.ground_truth for c in x.score]
    print("Ground truth:", sum(map(int, trues)) / len(trues))
    return x

fuzzing_scorer = FuzzingScorer(
    client,
    tokenizer=dataset.tokenizer,
    verbose=True,
    log_prob=False,
)
detection_scorer = DetectionScorer(
    client,
    tokenizer=dataset.tokenizer,
    verbose=True,
    log_prob=False,
)
#%%
import logging

from sae_auto_interp.logger import logger

logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.WARNING)
# top_level_await(explainer_pipe(record_first)).explanation
# top_level_await(dspy_explainer(record_first)).explanation
# top_level_await(default_explainer(record_first)).explanation
#%%
explainer_pipe = process_wrapper(
    # default_explainer,
    dspy_explainer,
    preprocess=explainer_preprocess,
    postprocess=explainer_postprocess,
)
scorer_pipe = process_wrapper(
    DSPyClassifier(
        fuzzing_scorer,
        # detection_scorer,
    ),
    preprocess=scorer_preprocess,
    postprocess=scorer_postprocess,
)
pipe = Pipeline(
    loader,
    explainer_pipe,
    scorer_pipe
)

# asyncio.run(pipe.run(1))
top_level_await(pipe.run(1))
# %%
