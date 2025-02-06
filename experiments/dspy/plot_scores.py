from matplotlib import pyplot as plt
import seaborn as sns
import json


class RandomColorDict(dict):
    def __missing__(self, key):
        self[key] = sns.color_palette("Paired", len(self) + 1)[len(self)]
        return self[key]


if __name__ == "__main__":
    with open("results/dspy_scores/accuracies.json") as accuracies_file:
        accuracies = json.load(accuracies_file)
    sns.set_theme()
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    kde_kwargs = dict(bw_method=0.25)
    colors = RandomColorDict()
    
    def plot_hist(ax, accs, label):
        sns.kdeplot(accs, ax=ax, **kde_kwargs, label=label, c=colors[label])
        acc = sum(accs) / len(accs)
        h = 5
        ax.plot([acc, acc], [0, h], c=colors[label], linewidth=2, linestyle="--")
        ax.scatter([acc], [h], c=colors[label])
    
    if "default_accuracies" in accuracies:
        plot_hist(axs[0, 0], accuracies["default_accuracies"], "default")
    if "default_accuracies_cot" in accuracies:
        plot_hist(axs[0, 0], accuracies["default_accuracies_cot"], "default_cot")
    axs[0, 0].set_title("Default pipeline")
    
    for name, acc in accuracies.get("dspy_accuracies", {}).items():
        print(name, sum(acc) / len(acc))
        plot_hist(axs[0, 1], acc, name)
        axs[0, 1].set_title(f"pure dspy")
    
    for name, acc in accuracies.get("dspy_explainer_accuracies", {}).items():
        plot_hist(axs[1, 0], acc, name)
        axs[1, 0].set_title(f"dspy explainer")
    
    for name, acc in accuracies.get("dspy_scorer_accuracies", {}).items():
        plot_hist(axs[1, 1], acc, name)
        axs[1, 1].set_title(f"dspy classifier")
    
    for ax in axs.flat:
        ax.set(xlabel='Accuracy', xlim=(0, 1))
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("results/dspy_scores.png")
