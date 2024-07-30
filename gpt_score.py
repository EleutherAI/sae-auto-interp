# %%

import os
import orjson
import pandas as pd

explanation_dir = "results/gpt2_top/gpt2_explanations"
recall_dir = "results/gpt2_top/gpt2_recall"
fuzz_dir = "results/gpt2_top/gpt2_fuzz"

scores = []

def _balanced_accuracy(data):

    score = 0

    for d in data:
        if d['prediction']:
            score += 1

    return score / len(data)


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

            scores.append({
                "layer": layer,
                "feature": feature,
                "explanation": explanation['explanation'],
                "recall": round(_balanced_accuracy(recall), 2),
                "fuzz": round(_balanced_accuracy(fuzz), 2),
                "combined": round((_balanced_accuracy(recall) + _balanced_accuracy(fuzz)) / 2, 2)
            })

        except Exception as e:

            scores.append({
                "layer": layer,
                "feature": feature,
                "explanation": "N/A",
                "recall": "-1",
                "fuzz": "-1",
                "combined": "-1"
            })

            continue

data = pd.DataFrame(scores)

data.to_csv("top_heatmap.csv", index=False)