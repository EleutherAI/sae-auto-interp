import asyncio
import json
import os
from functools import partial
from glob import glob
from pathlib import Path
from typing import Callable, cast

import orjson
import torch
from simple_parsing import ArgumentParser
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from delphi.clients import Offline, OpenRouter
from delphi.config import CacheConfig, ExperimentConfig, LatentConfig, RunConfig
from delphi.explainers import DefaultExplainer
from delphi.latents import LatentCache, LatentDataset
from delphi.latents.constructors import constructor
from delphi.latents.neighbours import NeighbourCalculator
from delphi.latents.samplers import sample
from delphi.latents.transforms import set_neighbours
from delphi.log.result_analysis import log_results
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import DetectionScorer, FuzzingScorer
from delphi.sparse_coders import load_hooks_sparse_coders
from delphi.utils import load_tokenized_data


def load_artifacts(run_cfg: RunConfig):
    if run_cfg.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = AutoModel.from_pretrained(
        run_cfg.model,
        device_map={"": "cuda"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=run_cfg.load_in_8bit)
            if run_cfg.load_in_8bit
            else None
        ),
        torch_dtype=dtype,
        token=run_cfg.hf_token,
    )

    hookpoint_to_sparse_encode = load_hooks_sparse_coders(model, run_cfg, compile=True)

    return run_cfg.hookpoints, hookpoint_to_sparse_encode, model


async def create_neighbours(
    latents_path: Path,
    neighbours_path: Path,
    hookpoints: list[str],
):
    """
    Creates a neighbours file for the given hookpoints.
    """
    neighbours_path.mkdir(parents=True, exist_ok=True)

    for hookpoint in hookpoints:
        neighbour_calculator = NeighbourCalculator(
            cache_dir=latents_path / hookpoint, number_of_neighbours=100
        )

        neighbour_calculator.populate_neighbour_cache(["co-occurrence"])

        neighbour_calculator.save_neighbour_cache(f"{neighbours_path}/{hookpoint}.json")


async def process_cache(
    latent_cfg: LatentConfig,
    run_cfg: RunConfig,
    experiment_cfg: ExperimentConfig,
    latents_path: Path,
    neighbours_path: Path,
    explanations_path: Path,
    scores_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_range: Tensor | None,
):
    """
    Converts SAE latent activations in on-disk cache in the `latents_path` directory
    to latent explanations in the `explanations_path` directory and explanation
    scores in the `fuzz_scores_path` directory.
    """
    explanations_path.mkdir(parents=True, exist_ok=True)

    fuzz_scores_path = scores_path / "fuzz"
    detection_scores_path = scores_path / "detection"
    fuzz_scores_path.mkdir(parents=True, exist_ok=True)
    detection_scores_path.mkdir(parents=True, exist_ok=True)

    if latent_range is None:
        latent_dict = None
    else:
        latent_dict = {
            hook: latent_range for hook in hookpoints
        }  # The latent range to explain
        latent_dict = cast(dict[str, Tensor], latent_dict)

    example_constructor = partial(
        constructor,
        n_not_active=experiment_cfg.n_non_activating,
        constructor_type=experiment_cfg.non_activating_source,
        ctx_len=experiment_cfg.example_ctx_len,
        max_examples=latent_cfg.max_examples,
    )
    sampler = partial(sample, cfg=experiment_cfg)
    if experiment_cfg.non_activating_source == "neighbours":
        neighbours = {}
        for hookpoint in hookpoints:
            with open(neighbours_path / f"{hookpoint}.json", "r") as f:
                neighbours[hookpoint] = json.load(f)["co-occurrence"]
        transform = partial(
            set_neighbours,
            neighbours=neighbours,
            threshold=0.0,
        )
    else:
        transform = None
    dataset = LatentDataset(
        raw_dir=str(latents_path),
        cfg=latent_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
        constructor=example_constructor,
        sampler=sampler,
        transform=transform,
    )

    if run_cfg.explainer_provider == "offline":
        client = Offline(
            run_cfg.explainer_model,
            max_memory=0.9,
            # Explainer models context length - must be able to accommodate the longest
            # set of examples
            max_model_len=run_cfg.explainer_model_max_len,
            num_gpus=run_cfg.num_gpus,
            statistics=run_cfg.verbose,
        )
    elif run_cfg.explainer_provider == "openrouter":
        if (
            "OPENROUTER_API_KEY" not in os.environ
            or not os.environ["OPENROUTER_API_KEY"]
        ):
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. Set "
                "`--explainer-provider offline` to use a local explainer model."
            )

        client = OpenRouter(
            run_cfg.explainer_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    else:
        raise ValueError(
            f"Explainer provider {run_cfg.explainer_provider} not supported"
        )

    def explainer_postprocess(result):
        with open(explanations_path / f"{result.record.latent}.txt", "wb") as f:
            f.write(orjson.dumps(result.explanation))
        return result

    explainer_pipe = process_wrapper(
        DefaultExplainer(
            client,
            tokenizer=dataset.tokenizer,
            threshold=0.3,
            verbose=run_cfg.verbose,
        ),
        postprocess=explainer_postprocess,
    )

    # Builds the record from result returned by the pipeline
    def scorer_preprocess(result):
        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.not_active
        return record

    # Saves the score to a file
    def scorer_postprocess(result, score_dir):
        safe_latent_name = str(result.record.latent).replace("/", "--")

        with open(score_dir / f"{safe_latent_name}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorer_pipe = Pipe(
        process_wrapper(
            DetectionScorer(
                client,
                tokenizer=dataset.tokenizer,  # type: ignore
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=False,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=detection_scores_path),
        ),
        process_wrapper(
            FuzzingScorer(
                client,
                tokenizer=dataset.tokenizer,  # type: ignore
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=False,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=fuzz_scores_path),
        ),
    )

    pipeline = Pipeline(
        dataset,
        explainer_pipe,
        scorer_pipe,
    )

    await pipeline.run(run_cfg.pipeline_num_proc)


def populate_cache(
    run_cfg: RunConfig,
    cfg: CacheConfig,
    model: PreTrainedModel,
    hookpoint_to_sparse_encode: dict[str, Callable],
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    """
    Populates an on-disk cache in `latents_path` with SAE latent activations.
    """
    latents_path.mkdir(parents=True, exist_ok=True)

    tokens = load_tokenized_data(
        cfg.ctx_len,
        tokenizer,
        cfg.dataset_repo,
        cfg.dataset_split,
        cfg.dataset_name,
        cfg.dataset_column,
        run_cfg.seed,
    )

    if run_cfg.filter_bos:
        if tokenizer.bos_token_id is None:
            print("Tokenizer does not have a BOS token, skipping BOS filtering")
        else:
            flattened_tokens = tokens.flatten()
            mask = ~torch.isin(flattened_tokens, torch.tensor([tokenizer.bos_token_id]))
            masked_tokens = flattened_tokens[mask]
            truncated_tokens = masked_tokens[
                : len(masked_tokens) - (len(masked_tokens) % cfg.ctx_len)
            ]
            tokens = truncated_tokens.reshape(-1, cfg.ctx_len)

    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=cfg.batch_size,
    )
    cache.run(cfg.n_tokens, tokens)

    cache.save_splits(
        # Split the activation and location indices into different files to make
        # loading faster
        n_splits=cfg.n_splits,
        save_dir=latents_path,
    )

    cache.save_config(save_dir=latents_path, cfg=cfg, model_name=run_cfg.model)


async def run(
    experiment_cfg: ExperimentConfig,
    latent_cfg: LatentConfig,
    cache_cfg: CacheConfig,
    run_cfg: RunConfig,
):
    base_path = Path.cwd() / "results"
    if run_cfg.name:
        base_path = base_path / run_cfg.name

    base_path.mkdir(parents=True, exist_ok=True)
    with open(base_path / "run_config.json", "w") as f:
        json.dump(run_cfg.__dict__, f, indent=4)

    latents_path = base_path / "latents"
    explanations_path = base_path / "explanations"
    scores_path = base_path / "scores"
    neighbours_path = base_path / "neighbours"
    visualize_path = base_path / "visualize"

    latent_range = torch.arange(run_cfg.max_latents) if run_cfg.max_latents else None

    hookpoints, hookpoint_to_sparse_encode, model = load_artifacts(run_cfg)
    tokenizer = AutoTokenizer.from_pretrained(run_cfg.model, token=run_cfg.hf_token)

    if (
        not glob(str(latents_path / ".*")) + glob(str(latents_path / "*"))
        or "cache" in run_cfg.overwrite
    ):
        populate_cache(
            run_cfg,
            cache_cfg,
            model,
            hookpoint_to_sparse_encode,
            latents_path,
            tokenizer,
        )
    else:
        print(f"Files found in {latents_path}, skipping cache population...")

    del model, hookpoint_to_sparse_encode
    if (
        not glob(str(neighbours_path / ".*")) + glob(str(neighbours_path / "*"))
        or "neighbours" in run_cfg.overwrite
    ):
        await create_neighbours(latents_path, neighbours_path, hookpoints)
    else:
        print(f"Files found in {neighbours_path}, skipping...")

    if (
        not glob(str(scores_path / ".*")) + glob(str(scores_path / "*"))
        or "scores" in run_cfg.overwrite
    ):
        await process_cache(
            latent_cfg,
            run_cfg,
            experiment_cfg,
            latents_path,
            neighbours_path,
            explanations_path,
            scores_path,
            hookpoints,
            tokenizer,
            latent_range,
        )
    else:
        print(f"Files found in {scores_path}, skipping...")

    if run_cfg.verbose:
        log_results(scores_path, visualize_path, run_cfg.hookpoints)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_cfg")
    parser.add_arguments(LatentConfig, dest="latent_cfg")
    parser.add_arguments(CacheConfig, dest="cache_cfg")
    parser.add_arguments(RunConfig, dest="run_cfg")
    args = parser.parse_args()

    asyncio.run(run(args.experiment_cfg, args.latent_cfg, args.cache_cfg, args.run_cfg))
