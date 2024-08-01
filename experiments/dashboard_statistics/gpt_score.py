# %%

import orjson
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

explanation_dir = "results/gpt2_explanations"
recall_dir = "results/gpt2_recall"
fuzz_dir = "results/gpt2_fuzz"

scores = []


def calculate_balanced_accuracy(data):
    ground_truths = [d["ground_truth"] for d in data]
    predictions = []

    for truth, d in zip(ground_truths, data):
        if truth:
            if d["prediction"]:
                predictions.append(True)
            else:
                predictions.append(False)

        else:
            if d["prediction"]:
                predictions.append(False)
            else:
                predictions.append(True)

    score = balanced_accuracy_score(ground_truths, predictions)
    return round(score, 2)


layers = range(0, 12, 2)
features = range(50)

for layer in layers:
    for feature in features:
        feature_name = f".transformer.h.{layer}_feature{feature}"

        try:
            with open(f"{explanation_dir}/{feature_name}.txt", "rb") as f:
                explanation = orjson.loads(f.read())

            with open(f"{recall_dir}/{feature_name}.txt", "rb") as f:
                recall = orjson.loads(f.read())

            with open(f"{fuzz_dir}/{feature_name}.txt", "rb") as f:
                fuzz = orjson.loads(f.read())

            scores.append(
                {
                    "layer": layer,
                    "feature": feature,
                    "explanation": explanation["explanation"],
                    "recall": round(calculate_balanced_accuracy(recall), 2),
                    "fuzz": round(calculate_balanced_accuracy(fuzz), 2),
                    "combined": round(
                        (
                            calculate_balanced_accuracy(recall)
                            + calculate_balanced_accuracy(fuzz)
                        )
                        / 2,
                        2,
                    ),
                }
            )

        except Exception:
            scores.append(
                {
                    "layer": layer,
                    "feature": feature,
                    "explanation": "N/A",
                    "recall": "-1",
                    "fuzz": "-1",
                    "combined": "-1",
                }
            )

            continue

data = pd.DataFrame(scores)

data.to_csv("random_heatmap.csv", index=False)
