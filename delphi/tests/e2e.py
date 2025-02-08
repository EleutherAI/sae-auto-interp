from pathlib import Path
import orjson
import torch
from torch import Tensor
import pandas as pd
import asyncio
import time

from delphi.config import ExperimentConfig, FeatureConfig, CacheConfig
from delphi.__main__ import run, RunConfig, load_artifacts


def parse_score_file(file_path):
    with open(file_path, "rb") as f:
        data = orjson.loads(f.read())

    return pd.DataFrame(
        [
            {
                "text": "".join(example["str_tokens"]),
                "distance": example["distance"],
                "ground_truth": example["ground_truth"],
                "prediction": example["prediction"],
                "probability": example["probability"],
                "correct": example["correct"],
                "activations": example["activations"],
                "highlighted": example["highlighted"],
            }
            for example in data
        ]
    )


def build_df(path: Path, target_modules: list[str], range: Tensor | None):
    accuracies = []
    probabilities = []
    score_types = []
    file_names = []
    feature_indices = []
    modules = []

    # Get subdirectories in the scores path
    scores_types = [d.name for d in path.iterdir() if d.is_dir()]
    print(scores_types)
    for score_type in scores_types:
        score_type_path = path / score_type
        for module in target_modules:
            for score_file in list(score_type_path.glob(f"*{module}*")) + list(
                score_type_path.glob(f".*{module}*")
            ):
                feature_idx = int(score_file.stem.split("feature")[-1])
                if range is not None and feature_idx not in range:
                    continue

                df = parse_score_file(score_file)

                # Calculate the accuracy and cross entropy loss for this feature
                file_names.append(score_file.stem)
                score_types.append(score_type)
                feature_indices.append(feature_idx)
                accuracies.append(df["correct"].mean())
                probabilities.append(df["probability"].mean())
                modules.append(module)

    df = pd.DataFrame(
        {
            "file_name": file_names,
            "score_type": score_types,
            "feature_idx": feature_indices,
            "accuracy": accuracies,
            "probability": probabilities,
            "module": modules,
        }
    )
    assert not df.empty
    return df


async def new_main():
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
    feature_cfg = FeatureConfig(
        width=32_768,
        min_examples=200,  # The minimum number of examples to consider for the feature to be explained
        max_examples=10_000,  # The maximum number of examples a feature may activate on before being excluded from explanation
    )
    run_cfg = RunConfig(
        overwrite=["cache", "scores", "visualize"],
        model="EleutherAI/pythia-160m",
        sparse_model="EleutherAI/sae-pythia-160m-32k",
        explainer_model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        hookpoints=["layers.3"],
        explainer_model_max_len=4208,
        max_features=100,
        seed=22,
        num_gpus=torch.cuda.device_count(),
        filter_tokens=None
    )

    start_time = time.time()
    await run(experiment_cfg, feature_cfg, cache_cfg, run_cfg)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    scores_path =  Path("results") / run_cfg.name / "scores"

    feature_range = torch.arange(run_cfg.max_features) if run_cfg.max_features else None
    hookpoints, submodule_to_sae, hooked_model, tokenizer = load_artifacts(run_cfg)
    del hooked_model, submodule_to_sae, tokenizer
    df = build_df(scores_path, hookpoints, feature_range)

    # Performs better than random guessing
    for score_type in df["score_type"].unique():
        print(df[df['score_type'] == score_type]["accuracy"].mean())
        assert df[df['score_type'] == score_type]["accuracy"].mean() > 0.55, f"Score type {score_type} has an accuracy of {df[df['score_type'] == score_type]['accuracy'].mean()}"


if __name__ == "__main__":
    asyncio.run(new_main())