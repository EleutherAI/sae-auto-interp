from pathlib import Path
import torch
import asyncio
import time


from delphi.config import ExperimentConfig, LatentConfig, CacheConfig
from delphi.__main__ import run, RunConfig
from delphi.log.result_analysis import build_scores_df, feature_balanced_score_metrics


async def test():
    cache_cfg = CacheConfig(
        dataset_repo="EleutherAI/rpj-v2-sample",
        dataset_split="train[:1%]",
        batch_size=8,
        ctx_len=256,
        n_splits=5,
        n_tokens=200_000,
    )
    experiment_cfg = ExperimentConfig(
        train_type="quantiles",
        test_type="quantiles",
        n_examples_train=40,
        n_examples_test=50,
    )
    latent_cfg = LatentConfig(
        width=32_768,
        min_examples=200,  # The minimum number of examples to consider for the latent to be explained
        max_examples=10_000,  # The maximum number of examples a latent may activate on before being excluded from explanation
    )
    run_cfg = RunConfig(
        name='test',
        overwrite=["cache", "scores"],
        model="EleutherAI/pythia-160m",
        sparse_model="EleutherAI/sae-pythia-160m-32k",
        explainer_model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        hookpoints=["layers.3"],
        explainer_model_max_len=4208,
        max_latents=100,
        seed=22,
        num_gpus=torch.cuda.device_count(),
        filter_bos=True
    )

    start_time = time.time()
    await run(experiment_cfg, latent_cfg, cache_cfg, run_cfg)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    # Performs better than random guessing
    scores_path =  Path("results") / run_cfg.name / "scores"
    df = build_scores_df(scores_path, run_cfg.hookpoints)
    for score_type in df["score_type"].unique():
        score_df = df[df['score_type'] == score_type]
        weighted_mean_metrics = feature_balanced_score_metrics(score_df, score_type)

        assert weighted_mean_metrics['accuracy'] > 0.55, f"Score type {score_type} has an accuracy of {weighted_mean_metrics['accuracy']}"


if __name__ == "__main__":
    asyncio.run(test())