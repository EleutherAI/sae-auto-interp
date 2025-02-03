from typing import cast
from functools import partial
from pathlib import Path
from argparse import ArgumentParser

import torch
from torch import Tensor
import orjson
import asyncio
from torchtyping import TensorType
from nnsight import LanguageModel
import lovely_tensors as lt
lt.monkey_patch()

from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.clients import Offline
from sae_auto_interp.autoencoders import load_gemma_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data, assert_type
from sae_auto_interp.scorers import FuzzingScorer


def process_cache(
    latents_path: Path,
    explanations_path: Path,
    fuzz_scores_path: Path,
):
    """
    Converts SAE feature activations in on-disk cache in the `latents_path` directory
    to feature explanations in the `explanations_path` directory and explanation
    scores in the `fuzz_scores_path` directory.
    """
    explanations_path.mkdir(parents=True, exist_ok=True)
    fuzz_scores_path.mkdir(parents=True, exist_ok=True)

    feature_cfg = FeatureConfig(
        width=131072,  # The number of latents of your SAE
        min_examples=200,  # The minimum number of examples to consider for the feature to be explained
        max_examples=10000,  # The maximum number of examples to be sampled from
        n_splits=5,  # How many splits was the cache split into
    )

    module = ".model.layers.10"  # The layer to explain
    feature_dict = {module: torch.arange(0, 131)}  # The latent range to explain
    feature_dict = cast(dict[str, int | Tensor], feature_dict)

    dataset = FeatureDataset(
        raw_dir=str(latents_path),  # The folder where the cache is stored
        cfg=feature_cfg,
        modules=[module],
        features=feature_dict,
    )

    experiment_cfg = ExperimentConfig(
        n_examples_train=40,  # Number of examples to sample for training
        n_examples_test=50,  # Number of examples to sample for testing
        example_ctx_len=256,  # Length of each example
        train_type="quantiles",  # Type of sampler to use for training.
        test_type="quantiles",  # Type of sampler to use for testing.
    )

    constructor = partial(
        default_constructor,
        tokens=dataset.tokens,
        n_random=experiment_cfg.n_random,
        ctx_len=experiment_cfg.example_ctx_len,
        max_examples=feature_cfg.max_examples,
    )
    sampler = partial(sample, cfg=experiment_cfg)
    loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)

    client = Offline(
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        max_memory=0.8,
        max_model_len=5120,
    )

    def explainer_postprocess(result):
        with open(explanations_path / f"{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.explanation))
        return result
        
    explainer_pipe = process_wrapper(
        DefaultExplainer(
            client,
            tokenizer=dataset.tokenizer,
        ),
        postprocess=explainer_postprocess,
    )
    
    # Builds the record from result returned by the pipeline
    def scorer_preprocess(result):
        record = result.record   
        record.explanation = result.explanation
        record.extra_examples = record.random_examples

        # Mutate the record to include test examples
        sample(record, experiment_cfg)
        
        # record.test
        # TODO understand why the processed activations (from default constructor) always have one non-zero 
        # items at 1 position and then all zeros elsewhere

        return record

    # Saves the score to a file
    def scorer_postprocess(result, score_dir):
        with open(score_dir / f"{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))


    scorer_pipe = process_wrapper(
        FuzzingScorer(client, tokenizer=dataset.tokenizer),
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir=fuzz_scores_path),
    )
    
    pipeline = Pipeline(
        loader,
        explainer_pipe,
        scorer_pipe,
    )
    number_of_parallel_latents = 10

    asyncio.run(pipeline.run(number_of_parallel_latents))


def populate_cache(latents_path: Path):
    """
    Populates an on-disk cache in `latents_path` with SAE feature activations.
    """
    latents_path.mkdir(parents=True, exist_ok=True)

    model = LanguageModel(
        "google/gemma-2-9b", device_map="cuda", dispatch=True, torch_dtype="float16"
    )
    submodule_to_sae, hooked_model = load_gemma_autoencoders(
        model, ae_layers=[10], average_l0s={10: 47}, size="131k", type="res"
    )
    hooked_model = assert_type(LanguageModel, hooked_model)

    cfg = CacheConfig(
        dataset_repo="EleutherAI/rpj-v2-sample",
        dataset_split="train[:1%]",
        batch_size=8,
        ctx_len=256,
        n_tokens=1_000_000,
        n_splits=5,
    )

    tokens = load_tokenized_data(
        ctx_len=cfg.ctx_len,
        tokenizer=hooked_model.tokenizer,
        dataset_repo=cfg.dataset_repo,
        dataset_split=cfg.dataset_split,
        dataset_name=cfg.dataset_name,
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

    # The config of the cache should be saved with the results such that it can be loaded later.
    cache.save_config(save_dir=str(latents_path), cfg=cfg, model_name="google/gemma-2-9b")


def visualize_feature_scores(
        fuzz_scores_path: Path,
        visualizations_path: Path,
    ):
    """
    Visualize the feature scores directory.
    """
    # Load the scores
    scores = []
    for score_file in fuzz_scores_path.glob("*.txt"):
        with open(score_file, "rb") as f:
            scores.append(orjson.loads(f.read()))
    breakpoint()


def main():
    latents_path = Path("results/latents")
    latents_path = Path("results/latents")
    explanations_path = Path("results/explanations")
    fuzz_scores_path = Path("results/scores/fuzz")
    visualizations_path = Path("results/visualizations")

    # n_splits = 5

    parser = ArgumentParser()
    parser.add_argument("--overwrite", nargs="+", choices=["cache", "scores", "visualize"], default=['scores', 'visualize'])
    args = parser.parse_args()

    if not latents_path.exists() or 'cache' in args.overwrite:
        populate_cache(latents_path)
    else:
        print("Cache already populated, skipping...")

    if not fuzz_scores_path.exists() or 'scores' in args.overwrite:
        process_cache(latents_path, explanations_path, fuzz_scores_path)
    else:
        print("Fuzz scores already computed, skipping...")

    if not visualizations_path.exists() or 'visualize' in args.overwrite:
        visualize_feature_scores(fuzz_scores_path, visualizations_path)
    else:
        print("Visualizations already computed, skipping...")


if __name__ == "__main__":
    main()
