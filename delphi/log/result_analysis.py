from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import plotly.express as px
import plotly.io as pio
from torch import Tensor

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469


def latent_balanced_score_metrics(
    df: pd.DataFrame, score_type: str, verbose: bool = True
):
    # Calculate weights based on non-errored examples
    valid_examples = df["total_examples"]
    weights = valid_examples / valid_examples.sum()

    metrics = {
        "accuracy": np.average(df["accuracy"], weights=weights),
        "f1_score": np.average(df["f1_score"], weights=weights),
        "precision": np.average(df["precision"], weights=weights),
        "recall": np.average(df["recall"], weights=weights),
        "false_positives": np.average(df["false_positives"], weights=weights),
        "false_negatives": np.average(df["false_negatives"], weights=weights),
        "true_positives": np.average(df["true_positives"], weights=weights),
        "true_negatives": np.average(df["true_negatives"], weights=weights),
        "positive_class_ratio": np.average(df["positive_class_ratio"], weights=weights),
        "negative_class_ratio": np.average(df["negative_class_ratio"], weights=weights),
        "total_positives": np.average(df["total_positives"], weights=weights),
        "total_negatives": np.average(df["total_negatives"], weights=weights),
        "true_positive_rate": np.average(df["true_positive_rate"], weights=weights),
        "true_negative_rate": np.average(df["true_negative_rate"], weights=weights),
        "false_positive_rate": np.average(df["false_positive_rate"], weights=weights),
        "false_negative_rate": np.average(df["false_negative_rate"], weights=weights),
        "single_token_ratio": (
            df["is_single_token"].mean() if "is_single_token" in df.columns else None
        ),
    }

    if verbose:
        print(f"\n--- {score_type.title()} Metrics ---")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1 Score: {metrics['f1_score']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")

        fractions_failed = [
            failed_count / (total_examples + failed_count)
            for failed_count, total_examples in zip(
                df["failed_count"], df["total_examples"]
            )
        ]
        print(
            f"""Average fraction of failed examples: \
{sum(fractions_failed) / len(fractions_failed):.3f}"""
        )

        if metrics["single_token_ratio"] is not None:
            print("\nSingle Token Features:")
            print(
                f"Ratio of single token features: {metrics['single_token_ratio']:.3f}"
            )

        print("\nConfusion Matrix:")
        print(f"True Positive Rate:  {metrics['true_positive_rate']:.3f}")
        print(f"True Negative Rate:  {metrics['true_negative_rate']:.3f}")
        print(f"False Positive Rate: {metrics['false_positive_rate']:.3f}")
        print(f"False Negative Rate: {metrics['false_negative_rate']:.3f}")

        print("\nClass Distribution:")
        print(
            f"""Positives: {df['total_positives'].sum():.0f} \
({metrics['positive_class_ratio']:.1%})"""
        )
        print(
            f"""Negatives: {df['total_negatives'].sum():.0f} \
({metrics['negative_class_ratio']:.1%})"""
        )
        print(f"Total: {df['total_examples'].sum():.0f}")

    return metrics


def parse_score_file(file_path):
    with open(file_path, "rb") as f:
        data = orjson.loads(f.read())
    is_single_token = data.get("is_single_token", False)
    df = pd.DataFrame(
        [
            {
                "text": "".join(example["str_tokens"]),
                "distance": example["distance"],
                "activating": example["activating"],
                "prediction": example["prediction"],
                "probability": example["probability"],
                "correct": example["correct"],
                "activations": example["activations"],
                "is_single_token": is_single_token,
            }
            for example in data
        ]
    )

    # Calculate basic counts
    failed_count = (df["prediction"].isna()).sum()
    df = df[df["prediction"].notna()]
    df.reset_index(drop=True, inplace=True)
    total_examples = len(df)
    total_positives = (df["activating"]).sum()
    total_negatives = (~df["activating"]).sum()

    # Calculate confusion matrix elements
    true_positives = ((df["prediction"] == 1) & (df["activating"])).sum()
    true_negatives = ((df["prediction"] == 0) & (~df["activating"])).sum()
    false_positives = ((df["prediction"] == 1) & (~df["activating"])).sum()
    false_negatives = ((df["prediction"] == 0) & (df["activating"])).sum()

    # Calculate rates
    true_positive_rate = true_positives / total_positives if total_positives > 0 else 0
    true_negative_rate = true_negatives / total_negatives if total_negatives > 0 else 0
    false_positive_rate = (
        false_positives / total_negatives if total_negatives > 0 else 0
    )
    false_negative_rate = (
        false_negatives / total_positives if total_positives > 0 else 0
    )

    # Calculate precision, recall, f1 (using sklearn for verification)
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = true_positive_rate  # Same as TPR
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Calculate accuracy
    accuracy = (
        (true_positives + true_negatives) / total_examples if total_examples > 0 else 0
    )

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
        "positive_class_ratio": (
            total_positives / total_examples if total_examples > 0 else 0
        ),
        "negative_class_ratio": (
            total_negatives / total_examples if total_examples > 0 else 0
        ),
        "failed_count": failed_count,
        "is_single_token": is_single_token,
    }

    for key, value in metrics.items():
        df.loc[0, key] = value

    return df


def build_scores_df(path: Path, target_modules: list[str], range: Tensor | None = None):
    metrics_cols = [
        "accuracy",
        "probability",
        "precision",
        "recall",
        "f1_score",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positive_rate",
        "true_negative_rate",
        "false_positive_rate",
        "false_negative_rate",
        "total_examples",
        "total_positives",
        "total_negatives",
        "positive_class_ratio",
        "negative_class_ratio",
        "failed_count",
        "is_single_token",
    ]
    df_data = {
        col: []
        for col in ["file_name", "score_type", "latent_idx", "module"] + metrics_cols
    }

    # Get subdirectories in the scores path
    scores_types = [d.name for d in path.iterdir() if d.is_dir()]

    for score_type in scores_types:
        score_type_path = path / score_type

        for module in target_modules:
            for score_file in list(score_type_path.glob(f"*{module}*")) + list(
                score_type_path.glob(f".*{module}*")
            ):
                if "latent" in score_file.stem:
                    latent_idx = int(score_file.stem.split("latent")[-1])
                else:
                    latent_idx = int(score_file.stem.split("feature")[-1])
                if range is not None and latent_idx not in range:
                    continue

                df = parse_score_file(score_file)

                # Calculate the accuracy and cross entropy loss for this latent
                df_data["file_name"].append(score_file.stem)
                df_data["score_type"].append(score_type)
                df_data["latent_idx"].append(latent_idx)
                df_data["module"].append(module)
                for col in metrics_cols:
                    df_data[col].append(df.loc[0, col])

    df = pd.DataFrame(df_data)
    assert not df.empty
    return df


def plot_line(df: pd.DataFrame, visualize_path: Path):
    visualize_path.mkdir(parents=True, exist_ok=True)

    for score_type in df["score_type"].unique():
        mask = df["score_type"] == score_type

        fig = px.histogram(
            df[mask],
            x="accuracy",
            title=f"Latent explanation accuracies for {score_type} scorer",
            nbins=100,
        )

        fig.write_image(visualize_path / f"{score_type}_accuracies.pdf", format="pdf")


def log_results(scores_path: Path, visualize_path: Path, target_modules: list[str]):
    df = build_scores_df(scores_path, target_modules)
    plot_line(df, visualize_path)

    for score_type in df["score_type"].unique():
        score_df = df[df["score_type"] == score_type]
        latent_balanced_score_metrics(score_df, score_type)
