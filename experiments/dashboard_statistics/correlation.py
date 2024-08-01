# %%
# Import necessary libraries
import os

import matplotlib.pyplot as plt
import numpy as np
import orjson
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score, r2_score

# Directory paths
explanation_dir_a = "results/gpt2_top/gpt2_explanations"
recall_dir_a = "results/gpt2_top/gpt2_recall"
fuzz_dir_a = "results/gpt2_top/gpt2_fuzz"

explanation_dir_b = "results/gpt2_explanations"
recall_dir_b = "results/gpt2_recall"
fuzz_dir_b = "results/gpt2_fuzz"

# Initialize a list to store scores
scores_a = []
scores_b = []


# Function to calculate balanced accuracy
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


# Function to extract layer and feature from file name
def to_feature(name):
    layer, feature = name.split("_")
    layer = layer.replace(".transformer.h.", "")
    feature = feature.replace("feature", "")
    return layer, feature


# Function to process directory
def process_directory(explanation_dir, recall_dir, fuzz_dir):
    scores = []
    for file in os.listdir(recall_dir):
        feature_name = file.replace(".txt", "")
        layer, feature = to_feature(feature_name)

        try:
            with open(f"{explanation_dir}/{file}", "rb") as f:
                explanation = orjson.loads(f.read())

            with open(f"{recall_dir}/{file}", "rb") as f:
                recall = orjson.loads(f.read())

            with open(f"{fuzz_dir}/{file}", "rb") as f:
                fuzz = orjson.loads(f.read())

            recall_score = calculate_balanced_accuracy(recall)
            fuzz_score = calculate_balanced_accuracy(fuzz)
            average_score = (recall_score + fuzz_score) / 2

            scores.append(
                {
                    "layer": layer,
                    "feature": feature,
                    "explanation": explanation["explanation"],
                    "recall": recall_score,
                    "fuzz": fuzz_score,
                    "average": average_score,
                }
            )

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    return scores


scores_a = process_directory(explanation_dir_a, recall_dir_a, fuzz_dir_a)
scores_b = process_directory(explanation_dir_b, recall_dir_b, fuzz_dir_b)


# df = pd.DataFrame(scores_a)
# df.to_csv('gpt2_top.csv', index=False)

# df = pd.DataFrame(scores_b)
# df.to_csv('gpt2_random.csv', index=False)

# %%
# Process both directories
scores_a = process_directory(explanation_dir_a, recall_dir_a, fuzz_dir_a)
scores_b = process_directory(explanation_dir_b, recall_dir_b, fuzz_dir_b)


# Function to plot data
def plot_data(ax, scores, title):
    recall_scores = [score["recall"] for score in scores]
    fuzz_scores = [score["fuzz"] for score in scores]
    average_scores = [score["average"] for score in scores]

    scatter = ax.scatter(
        recall_scores, fuzz_scores, c=average_scores, cmap="viridis", alpha=0.7
    )

    # Adding correlation line
    X = np.array(recall_scores).reshape(-1, 1)
    y = np.array(fuzz_scores)
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    ax.plot(recall_scores, y_pred, color="red", linewidth=2)

    # Calculate and display correlation and r^2
    correlation = np.corrcoef(recall_scores, fuzz_scores)[0, 1]
    r2 = r2_score(fuzz_scores, y_pred)
    ax.legend([f"Correlation: {correlation:.2f}\nR^2: {r2:.2f}"], loc="upper left")

    # Plot settings
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Fuzz")
    ax.set_xlim(0.3, 1)
    ax.set_ylim(0.3, 1)
    ax.grid(True)

    return scatter


# Plot the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

scatter1 = plot_data(ax1, scores_a, "Balanced accuracy from top twenty examples")
scatter2 = plot_data(ax2, scores_b, "Balanced accuracy from twenty random examples")

# Add colorbar
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.5)
cbar = plt.colorbar(scatter2, cax=cax, label="Average Score")

# Show plot
plt.tight_layout()
plt.show()
