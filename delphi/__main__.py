from typing import cast
from functools import partial
from pathlib import Path
from dataclasses import dataclass
from glob import glob
from dataclasses import dataclass
from multiprocessing import cpu_count
import asyncio
import re
import os

from simple_parsing import ArgumentParser, field
import torch
from torch import Tensor
import torch.nn as nn
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BitsAndBytesConfig,
)
import orjson
from torchtyping import TensorType
from nnsight import LanguageModel
from datasets import load_dataset
from sae.data import chunk_and_tokenize
from simple_parsing import field, list_field

from delphi.config import ExperimentConfig, FeatureConfig
from delphi.explainers import DefaultExplainer
from delphi.features import FeatureDataset, FeatureLoader
from delphi.features.constructors import default_constructor
from delphi.features.samplers import sample
from delphi.pipeline import Pipeline, process_wrapper
from delphi.clients import Offline, OpenRouter
from delphi.config import CacheConfig
from delphi.features import FeatureCache
from delphi.utils import assert_type
from delphi.scorers import FuzzingScorer, DetectionScorer
from delphi.pipeline import Pipe
from delphi.autoencoders.eleuther import load_and_hook_sparsify_models


@dataclass
class RunConfig:
    model: str = field(
        default="meta-llama/Meta-Llama-3-8B",
        positional=True,
    )
    """Name of the model to explain."""

    sparse_model: str = field(
        default="EleutherAI/sae-llama-3-8b-32x",
        positional=True,
    )
    """Name of the models associated with the model to explain, or path to
    directory containing its weights. Models must be loadable with sparsify."""

    hookpoints: list[str] = list_field()
    """List of hookpoints to load SAEs for."""

    explainer_model: str = field(
        default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    )
    """Name of the model to use for explanation generation."""

    explainer_model_max_len: int = field(
        default=5120,
    )
    """Maximum length of the explainer model context window."""

    explainer_provider: str = field(
        default="offline",
    )
    """Provider to use for explanation generation. Options are 'offline' for local models and 'openrouter' for API calls."""

    name: str = ""
    """The name of the run. Results will be saved in a directory with this name."""

    overwrite: list[str] = list_field()
    """Whether to overwrite existing parts of the run. Options are 'cache', 'scores', and 'visualize'."""

    max_features: int | None = None
    """Maximum number of features to explain for each SAE."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    hf_token: str | None = None
    """Huggingface API token for downloading models."""

    pipeline_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""

    num_gpus: int = field(
        default=1,
    )
    """Number of GPUs to use for explanation generation."""

    seed: int = field(
        default=22,
    )
    """Seed for the random number generator."""


def load_artifacts(run_cfg: RunConfig):
    if run_cfg.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = LanguageModel(
        run_cfg.model,
        device_map={"": "cuda"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=run_cfg.load_in_8bit)
            if run_cfg.load_in_8bit
            else None
        ),
        torch_dtype=dtype,
        token=run_cfg.hf_token,
        dispatch=True,
    )

    # Add SAE hooks to the model
    submodule_name_to_submodule, model = load_and_hook_sparsify_models(
        model,  # type: ignore
        run_cfg.sparse_model,
        run_cfg.hookpoints,
        k=run_cfg.max_features,
    )
    model = assert_type(LanguageModel, model)

    return run_cfg.hookpoints, submodule_name_to_submodule, model, model.tokenizer


async def process_cache(
    feature_cfg: FeatureConfig,
    run_cfg: RunConfig,
    experiment_cfg: ExperimentConfig,
    latents_path: Path,
    explanations_path: Path,
    scores_path: Path,
    # The layers to explain
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    feature_range: Tensor | None,
):
    """
    Converts SAE feature activations in on-disk cache in the `latents_path` directory
    to feature explanations in the `explanations_path` directory and explanation
    scores in the `fuzz_scores_path` directory.
    """
    explanations_path.mkdir(parents=True, exist_ok=True)

    fuzz_scores_path = scores_path / "fuzz"
    detection_scores_path = scores_path / "detection"
    fuzz_scores_path.mkdir(parents=True, exist_ok=True)
    detection_scores_path.mkdir(parents=True, exist_ok=True)

    if feature_range is None:
        feature_dict = None
    else:
        feature_dict = {
            hook: feature_range for hook in hookpoints
        }  # The latent range to explain
        feature_dict = cast(dict[str, int | Tensor], feature_dict)

    dataset = FeatureDataset(
        raw_dir=str(latents_path),
        cfg=feature_cfg,
        modules=hookpoints,
        features=feature_dict,
        tokenizer=tokenizer,
    )

    if run_cfg.explainer_provider == "offline":
        client = Offline(
            run_cfg.explainer_model,
            max_memory=0.8,
            # Explainer models context length - must be able to accomodate the longest set of examples
            max_model_len=run_cfg.explainer_model_max_len,
            num_gpus=run_cfg.num_gpus,
        )
    elif run_cfg.explainer_provider == "openrouter":
        if "OPENROUTER_API_KEY" not in os.environ or not os.environ["OPENROUTER_API_KEY"]:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. Set `--explainer-provider offline` to use a local explainer model."
            )

        client = OpenRouter(
            run_cfg.explainer_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    else:
        raise ValueError(
            f"Explainer provider {run_cfg.explainer_provider} not supported"
        )

    constructor = partial(
        default_constructor,
        token_loader=None,
        n_random=experiment_cfg.n_random,
        ctx_len=experiment_cfg.example_ctx_len,
        max_examples=feature_cfg.max_examples,
    )
    sampler = partial(sample, cfg=experiment_cfg)
    loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)

    def explainer_postprocess(result):
        with open(explanations_path / f"{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.explanation))
        return result

    explainer_pipe = process_wrapper(
        DefaultExplainer(
            client,
            tokenizer=dataset.tokenizer,
            threshold=0.3,
        ),
        postprocess=explainer_postprocess,
    )

    # Builds the record from result returned by the pipeline
    def scorer_preprocess(result):
        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.random_examples
        return record

    # Saves the score to a file
    def scorer_postprocess(result, score_dir):
        safe_feature_name = str(result.record.feature).replace("/", "--")

        with open(score_dir / f"{safe_feature_name}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorer_pipe = Pipe(
        process_wrapper(
            DetectionScorer(
                client,
                tokenizer=dataset.tokenizer,  # type: ignore
                batch_size=10,
                verbose=False,
                log_prob=False,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=detection_scores_path),
        ),
        process_wrapper(
            FuzzingScorer(
                client,
                tokenizer=dataset.tokenizer,  # type: ignore
                batch_size=10,
                verbose=False,
                log_prob=False,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=fuzz_scores_path),
        ),
    )

    pipeline = Pipeline(
        loader,
        explainer_pipe,
        scorer_pipe,
    )

    await pipeline.run(run_cfg.pipeline_num_proc)


def populate_cache(
    run_cfg: RunConfig,
    cfg: CacheConfig,
    hooked_model: LanguageModel,
    submodule_name_to_submodule: dict[str, nn.Module],
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    filter_tokens: Tensor | None,
):
    """
    Populates an on-disk cache in `latents_path` with SAE feature activations.
    """
    latents_path.mkdir(parents=True, exist_ok=True)

    data = load_dataset(
        cfg.dataset_repo, name=cfg.dataset_name, split=cfg.dataset_split
    )
    data = data.shuffle(run_cfg.seed)
    data = chunk_and_tokenize(
        data, tokenizer, max_seq_len=cfg.ctx_len, text_key=cfg.dataset_row
    )
    tokens = data["input_ids"]

    if filter_tokens is not None:
        flattened_tokens = tokens.flatten()
        mask = ~torch.isin(flattened_tokens, filter_tokens)
        masked_tokens = flattened_tokens[mask]
        truncated_tokens = masked_tokens[
            : len(masked_tokens) - (len(masked_tokens) % cfg.ctx_len)
        ]
        tokens = truncated_tokens.reshape(-1, cfg.ctx_len)

    tokens = cast(TensorType["batch", "seq"], tokens)

    cache = FeatureCache(
        hooked_model,
        submodule_name_to_submodule,
        batch_size=cfg.batch_size,
    )
    cache.run(cfg.n_tokens, tokens)

    cache.save_splits(
        # Split the activation and location indices into different files to make loading faster
        n_splits=cfg.n_splits,
        save_dir=latents_path,
    )

    cache.save_config(save_dir=str(latents_path), cfg=cfg, model_name=run_cfg.model)


async def run():
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_cfg")
    parser.add_arguments(FeatureConfig, dest="feature_cfg")
    parser.add_arguments(CacheConfig, dest="cache_cfg")
    parser.add_arguments(RunConfig, dest="run_cfg")
    args = parser.parse_args()

    base_path = Path("results")
    if args.run_cfg.name:
        base_path = base_path / args.run_cfg.name

    latents_path = base_path / "latents"
    explanations_path = base_path / "explanations"
    scores_path = base_path / "scores"

    feature_range = (
        torch.arange(args.run_cfg.max_features) if args.run_cfg.max_features else None
    )

    hookpoints, submodule_name_to_submodule, hooked_model, tokenizer = load_artifacts(
        args.run_cfg
    )

    if (
        not glob(str(latents_path / ".*")) + glob(str(latents_path / "*"))
        or "cache" in args.run_cfg.overwrite
    ):
        populate_cache(
            args.run_cfg,
            args.cache_cfg,
            hooked_model,
            submodule_name_to_submodule,
            latents_path,
            tokenizer,
            args.run_cfg.seed,
        )
    else:
        print(f"Files found in {latents_path}, skipping cache population...")

    del hooked_model, submodule_name_to_submodule

    if (
        not glob(str(scores_path / ".*")) + glob(str(scores_path / "*"))
        or "scores" in args.run_cfg.overwrite
    ):
        await process_cache(
            args.feature_cfg,
            args.run_cfg,
            args.experiment_cfg,
            latents_path,
            explanations_path,
            scores_path,
            hookpoints,
            tokenizer,
            feature_range,
        )
    else:
        print(f"Files found in {scores_path}, skipping...")


if __name__ == "__main__":
    asyncio.run(run())
