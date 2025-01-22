from matplotlib import pyplot as plt
import seaborn as sns
import json


if __name__ == "__main__":
    with open("results/dspy_scores/accuracies.json") as accuracies_file:
        accuracies = json.load(accuracies_file)
    sns.set_theme()
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    kde_kwargs = dict(bw_method=0.25)
    
    if "default_accuracies" in accuracies:
        sns.kdeplot(accuracies["default_accuracies"], ax=axs[0, 0], **kde_kwargs)
        axs[0, 0].set_title("Default pipeline")
    
    for name, acc in accuracies.get("dspy_accuracies", {}).items():
        sns.kdeplot(acc, ax=axs[0, 1], **kde_kwargs, label=name)
        axs[0, 1].set_title(f"pure dspy")
    
    for name, acc in accuracies.get("dspy_explainer_accuracies", {}).items():
        sns.kdeplot(acc, ax=axs[1, 0], **kde_kwargs, label=name)
        axs[1, 0].set_title(f"dspy explainer")
    
    for name, acc in accuracies.get("dspy_scorer_accuracies", {}).items():
        sns.kdeplot(acc, ax=axs[1, 1], **kde_kwargs, label=name)
        axs[1, 1].set_title(f"dspy classifier")
    
    for ax in axs.flat:
        ax.set(xlabel='Accuracy', xlim=(0, 1))
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("results/dspy_scores.png")
