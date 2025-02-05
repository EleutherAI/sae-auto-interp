from typing import cast
from functools import partial
from pathlib import Path
from dataclasses import dataclass
from glob import glob
from simple_parsing import ArgumentParser, field
import json

from datasets import load_dataset
from sae.data import chunk_and_tokenize
import torch
from torch import Tensor
import torch.nn as nn
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.io as pio
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
from sae_auto_interp.utils import assert_type
from sae_auto_interp.scorers import FuzzingScorer, DetectionScorer
from sae_auto_interp.pipeline import Pipe

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469


async def process_cache(
    latents_path: Path,
    explanations_path: Path,
    scores_path: Path,
    # The layers to explain
    modules: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    feature_range: Tensor,
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

    feature_cfg = FeatureConfig(
        width=131_072,  # The number of latents of your SAE
        min_examples=200,  # The minimum number of examples to consider for the feature to be explained
        max_examples=10_000,  # The maximum number of examples a feature may activate on before being excluded from explanation
    )

    feature_dict = {
        module: feature_range for module in modules
    }  # The latent range to explain
    feature_dict = cast(dict[str, int | Tensor], feature_dict)

    dataset = FeatureDataset(
        raw_dir=str(latents_path),
        cfg=feature_cfg,
        modules=modules,
        features=feature_dict,
        tokenizer=tokenizer,
    )

    experiment_cfg = ExperimentConfig()

    client = Offline(
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        max_memory=0.8,
        # Explainer models context length - must be able to accomodate the longest set of examples
        max_model_len=11_000,  
        num_gpus=8,
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
    number_of_parallel_latents = 11

    await pipeline.run(number_of_parallel_latents)


def populate_cache(
    hooked_model: LanguageModel,
    submodule_to_sae: dict[str, nn.Module],
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    """
    Populates an on-disk cache in `latents_path` with SAE feature activations.
    """
    latents_path.mkdir(parents=True, exist_ok=True)

    cfg = CacheConfig(
        dataset_repo="EleutherAI/rpj-v2-sample",
        dataset_split="train[:1%]",
        batch_size=8,
        ctx_len=256,
        n_splits=5,
        n_tokens=1_000_000,
    )

    SEED = 22
    data = load_dataset(cfg.dataset_repo, name=cfg.dataset_name, split=cfg.dataset_split)
    tokens_ds = chunk_and_tokenize(data, tokenizer, max_seq_len=cfg.ctx_len, text_key=cfg.dataset_row)
    tokens_ds = tokens_ds.shuffle(SEED)

    tokens = cast(TensorType["batch", "seq"], tokens_ds["input_ids"])

    mask = (tokens != 2).flatten()
    masked_tokens = tokens.flatten()[mask]
    truncated_tokens = masked_tokens[:len(masked_tokens) - (len(masked_tokens) % cfg.ctx_len)]
    tokens = truncated_tokens.reshape(-1, cfg.ctx_len)


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
    cache.save_config(
        save_dir=str(latents_path), cfg=cfg, model_name="google/gemma-2-9b"
    )


def parse_score_file(file_path):
    with open(file_path, "r") as f:
        content = f.read().strip()

    try:
        # Try direct JSON parsing first
        if content.startswith("[") and content.endswith("]"):
            data = json.loads(content)
        else:
            # Try XML format if direct JSON fails
            start = content.find("<document_content>") + len("<document_content>")
            end = content.find("</document_content>")
            if start == -1 or end == -1:
                print(f"Could not parse {file_path.name} - invalid format")
                return None
            data = json.loads(content[start:end])

        # Parse into DataFrame
        df = pd.DataFrame(
            [
                {
                    "text": "".join(segment["str_tokens"]),
                    "distance": segment["distance"],
                    "ground_truth": segment["ground_truth"],
                    "prediction": segment["prediction"],
                    "probability": segment["probability"],
                    "correct": segment["correct"],
                    "activations": segment["activations"],
                    "highlighted": segment["highlighted"],
                }
                for segment in data
            ]
        )

        return df
    except json.JSONDecodeError as e:
        print(f"Error parsing {file_path.name}: {e}")
        return None


def build_df(path: Path, target_modules: list[str], range: Tensor):
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
        # TODO update glob to only include files with a target_module in the name
        for module in target_modules:
            for score_file in list(score_type_path.glob(f"*{module}*")) + list(
                score_type_path.glob(f".*{module}*")
            ):
                feature_idx = int(score_file.stem.split("feature")[-1])
                if feature_idx not in range:
                    continue

                df = parse_score_file(score_file)
                if df is None:
                    continue

                # Calculate the accuracy and cross entropy loss for this example
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


def plot_line(df, visualizations_path: Path, target_modules: list[str]):
    visualizations_path.mkdir(parents=True, exist_ok=True)

    for score_type in df["score_type"].unique():
        # Create density curves for probabilities
        plot_data = []
        for module in target_modules:
            values = df[(df["score_type"] == score_type) & (df["module"] == module)][
                "probability"
            ]
            if len(values) > 0:
                if values.unique()[0] == 0:
                    print(f"Probabilities are all 0 for {module} in {score_type}")
                    continue
                kernel = stats.gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 200)
                density = kernel(x_range)
                plot_data.extend(
                    [
                        {"x": x, "density": d, "module": module}
                        for x, d in zip(x_range, density)
                    ]
                )

        if len(plot_data) > 0:
            fig = px.line(
                plot_data,
                x="x",
                y="density",
                color="module",
                title=f"Probability Distribution - {score_type}",
            )
            fig.write_image(
                visualizations_path / f"autointerp_probabilities_{score_type}.pdf",
                format="pdf",
            )

        # Create density curves for accuracies
        plot_data = []
        for module in target_modules:
            values = df[(df["score_type"] == score_type) & (df["module"] == module)][
                "accuracy"
            ]
            if len(values) > 0:
                kernel = stats.gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 200)
                density = kernel(x_range)
                plot_data.extend(
                    [
                        {"x": x, "density": d, "module": module}
                        for x, d in zip(x_range, density)
                    ]
                )

        if len(plot_data) > 0:
            fig = px.line(
                plot_data,
                x="x",
                y="density",
                color="module",
                title=f"Accuracy Distribution - {score_type}",
            )
            fig.write_image(
                visualizations_path / f"autointerp_accuracies_{score_type}.pdf",
                format="pdf",
            )


#     df.to_csv("autointerp_results.csv", index=False)

#     # Print statistics with inline formatting for debugger compatibility
#     for score_type in df["score_type"].unique():
#         for latent_type in df["latent_type"].unique():
#             print(f"{score_type} - {latent_type}:");
#             print(f"  Mean accuracy: {df[(df['score_type'] == score_type) & (df['latent_type'] == latent_type)]['accuracy'].mean()}");
#             print(f"  Mean probability: {df[(df['score_type'] == score_type) & (df['latent_type'] == latent_type)]['probabilities'].mean()}")



def load_artifacts():
    modules: list[str] = [".model.layers.10"]
    model = LanguageModel(
        "google/gemma-2-9b", device_map="cuda", dispatch=True, torch_dtype="float16"
    )
    model = assert_type(LanguageModel, model)
    submodule_to_sae, hooked_model = load_gemma_autoencoders(
        model, ae_layers=[10], average_l0s={10: 47}, size="131k", type="res"
    )
    hooked_model = assert_type(LanguageModel, hooked_model)

    tokenizer = hooked_model.tokenizer

    return modules, submodule_to_sae, hooked_model, tokenizer


@dataclass
class Args:
    """Whether to overwrite existing parts of the run. Options are 'cache', 'scores', and 'visualize'."""

    overwrite: list[str] = field(default_factory=lambda: ["scores", "visualize"])

    """The name of the run to be used in the output files. If not provided the run may overwrite existing files."""
    name: str = ""


async def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    base_path = Path("results")

    # Nest named runs in the results folder and do not overwrite
    if args.name:
        base_path = base_path / args.name

        if glob(str(base_path / "*.*"), recursive=True):
            raise ValueError(f"Run {args.name} already exists")

    latents_path = base_path / "latents"
    explanations_path = base_path / "explanations"
    scores_path = base_path / "scores"
    visualizations_path = base_path / "visualizations"

    feature_range = torch.arange(100)
    (modules, submodule_to_sae, hooked_model, tokenizer) = (
        load_artifacts()
    )

    if (
        not glob(str(latents_path / ".*")) + glob(str(latents_path / "*"))
        or "cache" in args.overwrite
    ):
        populate_cache(hooked_model, submodule_to_sae, latents_path, tokenizer)
    else:
        print(f"Files found in {latents_path}, skipping cache population...")

    del hooked_model, submodule_to_sae
    
    if (
        not glob(str(scores_path / ".*")) + glob(str(scores_path / "*"))
        or "scores" in args.overwrite
    ):
        await process_cache(
            latents_path,
            explanations_path,
            scores_path,
            modules,
            tokenizer,
            feature_range,
        )
    else:
        print(f"Files found in {scores_path}, skipping...")

    if (
        not glob(str(visualizations_path / ".*")) + glob(str(visualizations_path / "*"))
        or "visualize" in args.overwrite
    ):
        df = build_df(scores_path, modules, feature_range)
        plot_line(df, visualizations_path, modules)
    else:
        print(f"Files found in {visualizations_path}, skipping...")


if __name__ == "__main__":
    asyncio.run(main())
