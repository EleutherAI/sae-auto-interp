from pathlib import Path
import orjson
import torch
from torch import Tensor
import pandas as pd
import asyncio
import time
import numpy as np


from delphi.config import ExperimentConfig, FeatureConfig, CacheConfig
from delphi.__main__ import run, RunConfig

def parse_score_file(file_path):
    with open(file_path, "rb") as f:
        data = orjson.loads(f.read())
    
    df = pd.DataFrame([{
        "text": "".join(example["str_tokens"]),
        "distance": example["distance"],
        "ground_truth": example["ground_truth"],
        "prediction": example["prediction"],
        "probability": example["probability"],
        "correct": example["correct"],
        "activations": example["activations"],
        "highlighted": example["highlighted"]
    } for example in data])
    
    # Calculate basic counts
    failed_count = (df['prediction'] == -1).sum()
    df = df[df['prediction'] != -1]
    df.reset_index(drop=True, inplace=True)
    total_examples = len(df)
    total_positives = (df["ground_truth"]).sum()
    total_negatives = (~df["ground_truth"]).sum()
    
    # Calculate confusion matrix elements
    true_positives = ((df["prediction"] == 1) & (df["ground_truth"])).sum()
    true_negatives = ((df["prediction"] == 0) & (~df["ground_truth"])).sum()
    false_positives = ((df["prediction"] == 1) & (~df["ground_truth"])).sum()
    false_negatives = ((df["prediction"] == 0) & (df["ground_truth"])).sum()
    
    # Calculate rates
    true_positive_rate = true_positives / total_positives if total_positives > 0 else 0
    true_negative_rate = true_negatives / total_negatives if total_negatives > 0 else 0
    false_positive_rate = false_positives / total_negatives if total_negatives > 0 else 0
    false_negative_rate = false_negatives / total_positives if total_positives > 0 else 0
    
    # Calculate precision, recall, f1 (using sklearn for verification)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positive_rate  # Same as TPR
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate accuracy
    accuracy = (true_positives + true_negatives) / total_examples
    
    # Add metrics to first row
    metrics = {
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "total_examples": total_examples,
        "total_positives": total_positives,
        "total_negatives": total_negatives,
        "positive_class_ratio": total_positives / total_examples,
        "negative_class_ratio": total_negatives / total_examples,
        "failed_count": failed_count,
    }
    
    for key, value in metrics.items():
        df.loc[0, key] = value
    
    return df


def build_df(path: Path, target_modules: list[str], range: Tensor | None):
    metrics_cols = [
        "accuracy", "probability", "precision", "recall", "f1_score",
        "true_positives", "true_negatives", "false_positives", "false_negatives",
        "true_positive_rate", "true_negative_rate", "false_positive_rate", "false_negative_rate",
        "total_examples", "total_positives", "total_negatives",
        "positive_class_ratio", "negative_class_ratio", "failed_count"
    ]
    df_data = {
        col: [] 
        for col in ["file_name", "score_type", "feature_idx", "module"] + metrics_cols
    }

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
                df_data["file_name"].append(score_file.stem)
                df_data["score_type"].append(score_type)
                df_data["feature_idx"].append(feature_idx)
                df_data["module"].append(module)
                for col in metrics_cols: df_data[col].append(df.loc[0, col])


    df = pd.DataFrame(df_data)
    assert not df.empty
    return df


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
    feature_cfg = FeatureConfig(
        width=32_768,
        min_examples=200,  # The minimum number of examples to consider for the feature to be explained
        max_examples=10_000,  # The maximum number of examples a feature may activate on before being excluded from explanation
    )
    run_cfg = RunConfig(
        name='test',
        overwrite=["cache", "scores"],
        model="EleutherAI/pythia-160m",
        sparse_model="EleutherAI/sae-pythia-160m-32k",
        explainer_model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        hookpoints=["layers.3"],
        explainer_model_max_len=4208,
        max_features=100,
        seed=22,
        num_gpus=torch.cuda.device_count(),
        filter_bos=True
    )

    start_time = time.time()
    await run(experiment_cfg, feature_cfg, cache_cfg, run_cfg)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    scores_path =  Path("results") / run_cfg.name / "scores"

    feature_range = torch.arange(run_cfg.max_features) if run_cfg.max_features else None
    hookpoints, *_ = load_artifacts(run_cfg)

    df = build_df(scores_path, hookpoints, feature_range)

    # Performs better than random guessing
    for score_type in df["score_type"].unique():
        score_df = df[df['score_type'] == score_type]
        # Calculate weights based on non-errored examples
        valid_examples = score_df['total_examples']
        weights = valid_examples / valid_examples.sum()

        weighted_mean_metrics = {
            'accuracy': np.average(score_df['accuracy'], weights=weights),
            'f1_score': np.average(score_df['f1_score'], weights=weights),
            'precision': np.average(score_df['precision'], weights=weights),
            'recall': np.average(score_df['recall'], weights=weights),
            'false_positives': np.average(score_df['false_positives'], weights=weights),
            'false_negatives': np.average(score_df['false_negatives'], weights=weights),
            'true_positives': np.average(score_df['true_positives'], weights=weights),
            'true_negatives': np.average(score_df['true_negatives'], weights=weights),
            'positive_class_ratio': np.average(score_df['positive_class_ratio'], weights=weights),
            'negative_class_ratio': np.average(score_df['negative_class_ratio'], weights=weights),
            'total_positives': np.average(score_df['total_positives'], weights=weights),
            'total_negatives': np.average(score_df['total_negatives'], weights=weights),
            'true_positive_rate': np.average(score_df['true_positive_rate'], weights=weights),
            'true_negative_rate': np.average(score_df['true_negative_rate'], weights=weights),
            'false_positive_rate': np.average(score_df['false_positive_rate'], weights=weights),
            'false_negative_rate': np.average(score_df['false_negative_rate'], weights=weights),
        }

        print(f"\n=== {score_type.title()} Metrics ===")
        print(f"Accuracy: {weighted_mean_metrics['accuracy']:.3f}")
        print(f"F1 Score: {weighted_mean_metrics['f1_score']:.3f}")
        print(f"Precision: {weighted_mean_metrics['precision']:.3f}")
        print(f"Recall: {weighted_mean_metrics['recall']:.3f}")

        fractions_failed = [failed_count / total_examples for failed_count, total_examples in zip(score_df['failed_count'], score_df['total_examples'])]
        print(f"Average fraction of failed examples: {sum(fractions_failed) / len(fractions_failed):.3f}")

        print("\nConfusion Matrix:")
        print(f"True Positive Rate:  {weighted_mean_metrics['true_positive_rate']:.3f}")
        print(f"True Negative Rate:  {weighted_mean_metrics['true_negative_rate']:.3f}")
        print(f"False Positive Rate: {weighted_mean_metrics['false_positive_rate']:.3f}")
        print(f"False Negative Rate: {weighted_mean_metrics['false_negative_rate']:.3f}")
        
        print(f"\nClass Distribution:")
        print(f"Positives: {score_df['total_positives'].sum():.0f} ({weighted_mean_metrics['positive_class_ratio']:.1%})")
        print(f"Negatives: {score_df['total_negatives'].sum():.0f} ({weighted_mean_metrics['negative_class_ratio']:.1%})")
        print(f"Total: {score_df['total_examples'].sum():.0f}")

        assert weighted_mean_metrics['accuracy'] > 0.55, f"Score type {score_type} has an accuracy of {weighted_mean_metrics['accuracy']}"


if __name__ == "__main__":
    asyncio.run(test())