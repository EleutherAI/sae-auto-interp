# WARNING: This script is out of date and doesn't run.
# `delphi.__main__` is more complete and demonstrates correct usage of `delphi`.

import asyncio
import json
import os
import time
from functools import partial

import orjson
import torch
from simple_parsing import ArgumentParser

from delphi.clients import Offline
from delphi.config import ExperimentConfig, LatentConfig
from delphi.explainers import DefaultExplainer
from delphi.latents import LatentDataset, LatentLoader
from delphi.latents.constructors import default_constructor
from delphi.latents.samplers import sample
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import DetectionScorer, FuzzingScorer

# run with python examples/example_script.py --model gemma/16k --module
# .model.layers.10 --latents 100 --experiment_name test


def main(args):
    module = args.module
    latent_cfg = args.latent_options
    experiment_cfg = args.experiment_options
    shown_examples = args.shown_examples
    n_latents = args.latents
    start_latent = args.start_latent
    sae_model = args.model
    latent_dict = {f"{module}": torch.arange(start_latent, start_latent + n_latents)}
    dataset = LatentDataset(
        raw_dir="raw_latents",
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

    client = Offline(
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        max_memory=0.8,
        max_model_len=5120,
    )

    ### Build Explainer pipe ###
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
        postprocess=explainer_postprocess,
    )

    # save the experiment config
    with open(
        f"results/explanations/{sae_model}/{experiment_name}/experiment_config.json",
        "w",
    ) as f:
        print(experiment_cfg.to_dict())
        f.write(json.dumps(experiment_cfg.to_dict()))

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

    scorer_pipe = Pipe(
        process_wrapper(
            DetectionScorer(
                client,
                tokenizer=dataset.tokenizer,
                batch_size=shown_examples,
                verbose=False,
                log_prob=True,
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
                log_prob=True,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="fuzz"),
        ),
    )

    ### Build the pipeline ###

    pipeline = Pipeline(
        loader,
        explainer_pipe,
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
    parser.add_arguments(ExperimentConfig, dest="experiment_options")
    parser.add_arguments(LatentConfig, dest="latent_options")
    args = parser.parse_args()
    experiment_name = args.experiment_name

    main(args)
