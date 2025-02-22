from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
from glob import glob
import json
import os


sns.set_theme()
pkm_score_dir = Path("results/scores/itda_cache")
for config_group in [
    ("pythia-l9_mlp-transcoder-mean-skip-k32", "pythia-l9_mlp-mean-k32")
    # ("with_pkm_sae", "without_pkm_sae"),
    # ("with_pkm_transcoder", "without_pkm_transcoder"),
]:
    for layer in (9,):
        fuzz_accs = {}
        detect_accs = {}
        group_name = "-".join(config_group)
        for config_name in config_group:
            config_dir = pkm_score_dir / config_name
            def feature_accs(method):
                score_dir = config_dir / "default" / method
                feature_accs = []
                for s in os.listdir(score_dir):
                    if not s.endswith(".txt"):
                        continue
                    if f"layers.{layer}" not in s:
                        continue
                    try:
                        data = json.load(open(os.path.join(score_dir, s)))
                    except ValueError:
                        print("Error parsing", os.path.join(score_dir, s))
                        continue
                    corrects = []
                    for text in data:
                        corrects.append(int(text["correct"]))
                    feature_accs.append(sum(corrects)/len(corrects))
                if not feature_accs:
                    raise FileNotFoundError
                return feature_accs
            try:
                fuzz_accs[config_name] = feature_accs("fuzz")
                detect_accs[config_name] = feature_accs("detection")
            except FileNotFoundError:
                print(f"Skipping layer {layer} for config {group_name}")
                break
        else:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].set(xlabel="Accuracy", xlim=(0, 1), ylabel="Number of features", title="Fuzz")
            axs[1].set(xlabel="Accuracy", xlim=(0, 1), ylabel="Number of features", title="Detect")
            for config_name in config_group:
                sns.histplot(fuzz_accs[config_name], bins=20, alpha=0.5, ax=axs[0], label=config_name)
                sns.histplot(detect_accs[config_name], bins=20, alpha=0.5, ax=axs[1], label=config_name)

            axs[0].legend()
            fig.suptitle(f"{group_name} Layer {layer}, both routers")
            save_dir = f"results/itda_autointerp/{group_name}"
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f"{save_dir}/{layer}.png")