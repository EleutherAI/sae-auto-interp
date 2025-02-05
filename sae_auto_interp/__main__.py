from typing import cast
from functools import partial
from pathlib import Path
from dataclasses import dataclass
from glob import glob
from blinker import Namespace
from simple_parsing import ArgumentParser, field
import asyncio
import torch
from torch import Tensor
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import orjson
from torchtyping import TensorType
from nnsight import LanguageModel

from dataclasses import dataclass
from multiprocessing import cpu_count

import torch
from torch import Tensor
from simple_parsing import field, list_field
from transformers import AutoModel, BitsAndBytesConfig

from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.clients import Offline
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data, assert_type
from sae_auto_interp.scorers import FuzzingScorer, DetectionScorer
from sae_auto_interp.pipeline import Pipe

from sae.sae import Sae


@dataclass
class RunConfig():
    model: str = field(
        default="meta-llama/Meta-Llama-3-8B",
        positional=True,
    )
    """Name of the model to explain."""

    sae: str = field(
        default="EleutherAI/sae-llama-3-8b-32x",
        positional=True,
    )
    """Name of the SAEs associated with the model to explain."""

    explainer_model: str = field(
        default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        positional=True,
    )
    """Name of the model to use for explanation generation."""

    name: str = ""
    """The name of the run. If not provided the run may overwrite existing files."""

    overwrite: list[str] = list_field()
    """Whether to overwrite existing parts of the run. Options are 'cache', 'scores', and 'visualize'."""

    hookpoints: list[str] = list_field()
    """List of hookpoints to load SAEs for."""

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



def load_artifacts(args):
    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = AutoModel.from_pretrained(
        args.model,
        device_map={"": "cuda"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=args.load_in_8bit)
            if args.load_in_8bit
            else None
        ),
        torch_dtype=dtype,
        token=args.hf_token,
    )

    model = LanguageModel(model, device_map="cuda")

    # Add SAE hooks to the model
    submodule_to_sae = {}
    if len(args.hookpoints) == 1:
        sae = Sae.load_from_hub(args.sae, hookpoint=args.hookpoints[0])
        submodule_to_sae[args.hookpoints[0]] = sae
    else:
        saes = Sae.load_many(args.sae)
        for hookpoint in args.hookpoints:
            submodule_to_sae[hookpoint] = saes[hookpoint]

    model = assert_type(LanguageModel, model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    return args.hookpoints, submodule_to_sae, model, tokenizer

async def process_cache(
    args,
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

    feature_dict = {
        hook: feature_range for hook in hookpoints
    }  # The latent range to explain
    feature_dict = cast(dict[str, int | Tensor], feature_dict)

    dataset = FeatureDataset(
        raw_dir=str(latents_path),
        cfg=args.feature_cfg,
        modules=hookpoints,
        features=feature_dict,
        tokenizer=tokenizer,
    )

    client = Offline(
        args.run_cfg.explainer_model,
        max_memory=0.8,
        # Explainer models context length - must be able to accomodate the longest set of examples
        max_model_len=8192,
        num_gpus=args.run_cfg.num_gpus,
    )

    constructor = partial(
        default_constructor,
        token_loader=None,
        n_random=args.experiment_cfg.n_random,
        ctx_len=args.experiment_cfg.example_ctx_len,
        max_examples=args.feature_cfg.max_examples,
    )
    sampler = partial(sample, cfg=args.experiment_cfg)
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

    await pipeline.run(args.run_cfg.pipeline_num_proc)


def populate_cache(
    args,
    hooked_model: LanguageModel,
    submodule_to_sae: dict[str, nn.Module],
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    """
    Populates an on-disk cache in `latents_path` with SAE feature activations.
    """
    latents_path.mkdir(parents=True, exist_ok=True)

    cfg = args.cache_cfg

    tokens = load_tokenized_data(
        ctx_len=cfg.ctx_len,
        tokenizer=tokenizer,
        dataset_repo=cfg.dataset_repo,
        dataset_split=cfg.dataset_split,
        dataset_name=cfg.dataset_name,
        add_bos_token=False,
    )
    tokens = cast(TensorType["batch", "seq"], tokens)

    cache = FeatureCache(
        hooked_model,
        submodule_to_sae,
        batch_size=cfg.batch_size,
    )
    cache.run(cfg.n_tokens, tokens)

    cache.save_splits(
        # Split the activation and location indices into different files to make loading faster
        n_splits=cfg.n_splits,
        save_dir=latents_path,
    )

    cache.save_config(
        save_dir=str(latents_path), cfg=cfg, model_name=args.model
    )

async def run():
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_cfg")
    parser.add_arguments(FeatureConfig, dest="feature_cfg")
    parser.add_arguments(CacheConfig, dest="cache_cfg")
    parser.add_arguments(RunConfig, dest="run_cfg")    
    args = parser.parse_args()

    feature_range = torch.arange(args.max_features) if args.max_features else None
    
    hookpoints, submodule_to_sae, hooked_model, tokenizer = load_artifacts(args)

    base_path = Path("results")
    if args.name:
        base_path = base_path / args.name

    latents_path = base_path / "latents"
    explanations_path = base_path / "explanations"
    scores_path = base_path / "scores"

    if (
        not glob(str(latents_path / ".*")) + glob(str(latents_path / "*"))
        or "cache" in args.overwrite
    ):
        populate_cache(args, hooked_model, submodule_to_sae, latents_path, tokenizer)
    else:
        print(f"Files found in {latents_path}, skipping cache population...")

    del hooked_model, submodule_to_sae
    
    if (
        not glob(str(scores_path / ".*")) + glob(str(scores_path / "*"))
        or "scores" in args.overwrite
    ):
        await process_cache(
            args,
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
