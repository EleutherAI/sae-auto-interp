from matplotlib import pyplot as plt
import seaborn as sns
import json


if __name__ == "__main__":
    with open("results/dspy_scores/accuracies.json") as accuracies_file:
        accuracies = json.load(accuracies_file)
    sns.set_theme()
    if "default_accuracies" in accuracies:
        sns.kdeplot(accuracies["default_accuracies"], label="Default pipeline")
    for name, acc in accuracies.get("dspy_accuracies", {}).items():
        sns.kdeplot(acc, label=f"pure dspy \"{name}\"")
    for name, acc in accuracies.get("dspy_explainer_accuracies", {}).items():
        sns.kdeplot(acc, label=f"dspy explainer \"{name}\"")
    for name, acc in accuracies.get("dspy_classifier_accuracies", {}).items():
        sns.kdeplot(acc, label=f"dspy classifier \"{name}\"")
    plt.xlabel("Accuracy")
    plt.xlim(0, 1)
    plt.legend()
    plt.savefig("results/dspy_scores.png")
