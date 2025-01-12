#%%
# VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=6,7 vllm serve "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --tensor-parallel-size 2 --enforce-eager
# CUDA_VISIBLE_DEVICES=6,7 python -m sglang.launch_server --model-path  "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --port 8000 --host 0.0.0.0 --tensor-parallel-size=2 --mem-fraction-static=0.5

import os
import sys

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
import logging
import os
from functools import partial

import dotenv
import nest_asyncio
import orjson
import torch
from dspy import LM

from sae_auto_interp.clients import DSPy
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer, DSPyExplainer
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.scorers import DetectionScorer, DSPyClassifier, FuzzingScorer

nest_asyncio.apply()
logging.basicConfig(level=logging.WARNING)

#%%
# LM_PROVIDER = "openrouter"
LM_PROVIDER = "vllm"
USE_BIG_LLAMA = True
#%%
def top_level_await(fn):
    # https://stackoverflow.com/a/61331974
    try:
        return asyncio.get_running_loop().run_until_complete(fn)
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        return asyncio.run(fn)


environ = dotenv.dotenv_values(os.getcwd() + "/.env")
if LM_PROVIDER == "openrouter":
    or_model = (
        "meta-llama/llama-3.3-70b-instruct"
        if USE_BIG_LLAMA else
        # "meta-llama/llama-3.1-8b-instruct"
        "meta-llama/llama-3-8b-instruct"
    )
    dspy_lm = LM(
        # "openrouter/" + or_model,
        "openrouter/" + or_model,
        api_key=environ["OPENROUTER_API_KEY"],
        num_retries=16,
        # api_base="https://openrouter.ai/v1/",
    )
    client = DSPy(dspy_lm)
    # from sae_auto_interp.clients import OpenRouter
    # client = OpenRouter(or_model, api_key=environ["OPENROUTER_API_KEY"])
elif LM_PROVIDER == "vllm":
    assert USE_BIG_LLAMA
    dspy_lm = LM(
        "openai/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        api_base="http://localhost:8000/v1/",
        api_key="u"
    )
    client = DSPy(dspy_lm)
elif LM_PROVIDER == "groq":
    dspy_lm = LM(
        "llama-3.3-70b-specdec" if USE_BIG_LLAMA else "llama-3.1-8b-instant",
        api_key=environ["GROQ_API_KEY"],
        api_base="https://api.groq.com/openai/v1/",
    )
    client = DSPy(dspy_lm)
prompt = [
    {"role": "user", "content": "What is the capital of France?"},
]
# print(await client.generate(prompt))
print(top_level_await(client.generate(prompt)))
#%%
LOAD_PYTHIA = False

if LOAD_PYTHIA:
    # cache_dir = "/mnt/ssd-1/gpaulo/SAE-Zoology/extras/transcoders" \
    #             "/raw_features/pythia_pile/SAE-2-seed-4k"
    cache_dir = "/mnt/ssd-1/gpaulo/SAE-Zoology/extras/transcoders" \
                "/raw_features/pythia_pile/SkipTranscoder"
    module = ".gpt_neox.layers.6.mlp"
    sae_model = "pythia_pile-skiptranscoder"
else:
    cache_dir = "../raw_features"
    module = ".model.layers.10"
    sae_model = "gemma/16k"

if LOAD_PYTHIA:
    start_feature = 0
    n_features = 16
else:
    start_feature = 13107
    # start_feature = 0
    n_features = 48
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
    n_examples_train=50,
    n_examples_test=50,
    n_quantiles=10,
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
default_explainer = DefaultExplainer(
    client,
    tokenizer=dataset.tokenizer,
    threshold=0.2,
    activations=True,
)
dspy_explainer = DSPyExplainer(
    client.client,
    tokenizer=dataset.tokenizer,
    verbose=True,
)

fuzzing_scorer = FuzzingScorer(
    client,
    tokenizer=dataset.tokenizer,
    verbose=True,
    log_prob=False,
    batch_size=5,
)
detection_scorer = DetectionScorer(
    client,
    tokenizer=dataset.tokenizer,
    verbose=True,
    log_prob=False,
    batch_size=5,
)
#%%

# logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.WARNING)
#%%
experiment_name = "example_dspy"
results_suffix = f"{sae_model}{module}/{experiment_name}"
results_dir = "../results"
os.makedirs(f"{results_dir}/explanations/{results_suffix}", exist_ok=True)

def explainer_preprocess(x):
    return x
def explainer_postprocess(result):
    with open(
        f"{results_dir}/explanations/{results_suffix}/{result.record.feature}.txt",
        "wb",
    ) as f:
        f.write(orjson.dumps(result.explanation))

    return result

def scorer_preprocess(result):
    record = result.record
    record.explanation = result.explanation
    record.extra_examples = record.random_examples
    print("Explanation for scorer:", record.explanation)
    return record
def scorer_postprocess(x):
    corrects = [c.correct for c in x.score]
    print("Accuracy:", sum(map(int, corrects)) / len(corrects))
    # trues = [c.ground_truth for c in x.score]
    # print("Ground truth:", sum(map(int, trues)) / len(trues))
    # return x
    return list(map(int, corrects))


explainer_pipe = process_wrapper(
    default_explainer,
    # dspy_explainer,
    preprocess=explainer_preprocess,
    postprocess=explainer_postprocess,
)
scorer_pipe = process_wrapper(
    # DSPyClassifier(
        fuzzing_scorer,
        # detection_scorer,
    # ),
    preprocess=scorer_preprocess,
    postprocess=scorer_postprocess,
)
pipe = Pipeline(
    loader,
    explainer_pipe,
    scorer_pipe
)

corrects = top_level_await(pipe.run(80))
sum(map(sum, corrects)) / sum(map(len, corrects))
# %%
