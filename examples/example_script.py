import asyncio
import json
import os
import time
from pathlib import Path
from functools import partial

import orjson
import torch
from simple_parsing import ArgumentParser

from delphi.clients import Offline, OpenRouter
from delphi.config import ExperimentConfig, LatentConfig
from delphi.explainers import DefaultExplainer
from delphi.latents import LatentDataset, LatentLoader
from delphi.latents.constructors import default_constructor
from delphi.latents.samplers import sample
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import DetectionScorer, FuzzingScorer

"""
uv run python -m examples.example_script --model monet_cache_converted/850m --module .model.layers.4.router --latents 6144  --width 262144
uv run python -m sglang_router.launch_server --model-path "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --port 8000 --host 0.0.0.0 --tensor-parallel-size=2 --mem-fraction-static=0.8 --dp-size 2 
uv run python -m examples.example_script --model itda_cache/pythia-l9_mlp-transcoder-mean-skip-k32 --module gpt_neox.layers.9.mlp --latents 500 --width 50142 --random_subset
uv run python -m examples.example_script --model pkm_saes/baseline --module model.layers.9 --latents 500 --random_subset
"""

# run with python examples/example_script.py --model gemma/16k --module
# .model.layers.10 --latents 100 --experiment_name test


async def main(args):
    module = args.module
    latent_cfg = args.latent_options
    experiment_cfg = args.experiment_options
    shown_examples = args.shown_examples
    n_latents = args.latents
    start_latent = args.start_latent
    sae_model = args.model

    raw_dir = f"results/{args.model}"
    features = torch.arange(start_latent,start_latent+n_latents)
    cache_config_dir = f"{raw_dir}/{module}/config.json"
    with open(cache_config_dir, "r") as f:
        cache_config = json.load(f)
    if "width" in cache_config:
        latent_cfg.width = cache_config["width"]
    
    max_feat = 0
    for st_file in (Path(raw_dir) / module).glob(f"*.safetensors"):
        _, end = map(int, st_file.stem.split("_"))
        max_feat = max(max_feat, end)
    if max_feat != 0:
        latent_cfg.width = max_feat + 1
    
    if args.random_subset:
        torch.manual_seed(0)
        features = torch.randperm(latent_cfg.width)[:n_latents]
    latent_dict = {f"{module}": features}

    dataset = LatentDataset(
        raw_dir=raw_dir,
        cfg=latent_cfg,
        modules=[module],
        latents=latent_dict,
    )

    constructor = partial(
        default_constructor,
        token_loader=lambda: dataset.load_tokens(),
        n_random=experiment_cfg.n_random,
        ctx_len=experiment_cfg.example_ctx_len,
        max_examples=latent_cfg.max_examples,
    )
    sampler = partial(sample, cfg=experiment_cfg)
    loader = LatentLoader(dataset, constructor=constructor, sampler=sampler)
    ### Load client ###
    
    # client = Offline("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",max_memory=0.8,max_model_len=5120)
    client = OpenRouter("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", api_key="hey",
                        base_url="http://localhost:8000/v1/chat/completions")
    
    ### Build Explainer pipe ###
    def explainer_preprocess(record):
        explanation_path = f"results/explanations/{sae_model}/{experiment_name}/{record.feature}.txt"
        if os.path.exists(explanation_path):
            return ExplainerResult(record=record,
                                   explanation=orjson.loads(open(explanation_path, "rb").read()))
        return record
    
    def explainer_postprocess(result):
        with open(
            f"results/explanations/{sae_model}/{experiment_name}/{result.record.latent}.txt",
            "wb",
        ) as f:
            f.write(orjson.dumps(result.explanation))

        return result

    # try making the directory if it doesn't exist
    os.makedirs(f"results/explanations/{sae_model}/{experiment_name}", exist_ok=True)

    explainer_pipe = process_wrapper(
        DefaultExplainer(
            client,
            tokenizer=dataset.tokenizer,
            threshold=0.3,
        ),
        preprocess=explainer_preprocess,
        postprocess=explainer_postprocess,
    )

    # save the experiment config
    with open(
        f"results/explanations/{sae_model}/{experiment_name}/experiment_config.json",
        "w",
    ) as f:
        print(experiment_cfg.to_dict())
        f.write(json.dumps(experiment_cfg.to_dict()))

    pipeline = Pipeline(
        loader,
        explainer_pipe,
    )
    explanations = await pipeline.run(100)

    ### Build Scorer pipe ###

    def scorer_preprocess(result):
        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.not_active

        return record

    def scorer_postprocess(result, score_dir):
        record = result.record
        with open(
            f"results/scores/{sae_model}/{experiment_name}/{score_dir}/{record.latent}.txt",
            "wb",
        ) as f:
            f.write(orjson.dumps(result.score))

    os.makedirs(
        f"results/scores/{sae_model}/{experiment_name}/detection", exist_ok=True
    )
    os.makedirs(f"results/scores/{sae_model}/{experiment_name}/fuzz", exist_ok=True)

    # save the experiment config
    with open(
        f"results/scores/{sae_model}/{experiment_name}/detection/experiment_config.json",
        "w",
    ) as f:
        f.write(json.dumps(experiment_cfg.to_dict()))

    with open(
        f"results/scores/{sae_model}/{experiment_name}/fuzz/experiment_config.json", "w"
    ) as f:
        f.write(json.dumps(experiment_cfg.to_dict()))

    log_prob = False
    scorer_pipe = Pipe(process_wrapper(
            DetectionScorer(
                client,
                tokenizer=dataset.tokenizer,
                batch_size=shown_examples,
                verbose=False,
                log_prob=log_prob
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="detection"),
        ),
        process_wrapper(
            FuzzingScorer(
                client,
                tokenizer=dataset.tokenizer,
                batch_size=shown_examples,
                verbose=False,
                log_prob=log_prob
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="fuzz"),
        ),
    )

    ### Build the pipeline ###

    pipeline = Pipeline(
        explanations,
        scorer_pipe,
    )
    start_time = time.time()
    asyncio.run(pipeline.run(50))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--shown_examples", type=int, default=5)
    parser.add_argument("--model", type=str, default="gemma/16k")
    parser.add_argument("--module", type=str, default=".model.layers.10")
    parser.add_argument("--latents", type=int, default=100)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--random_subset", action="store_true")
    parser.add_argument("--neighbors", action="store_true")
    parser.add_argument("--substitute_within_group_explanations", action="store_true")
    parser.add_arguments(ExperimentConfig, dest="experiment_options")
    parser.add_arguments(LatentConfig, dest="latent_options")
    args = parser.parse_args()
    experiment_name = args.experiment_name
    


    asyncio.run(main(args))