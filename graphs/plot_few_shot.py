# %%
import os
import json
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

directories = [
    f"scores/fuzz_70b/simple_local_70b_nfew{few_shots}" 
    for few_shots in range(2, 9)
]

directories.append(
    "scores/fuzz_70b/simple_local_70b_q4_nt5"
)


def load_data(directory):
    data = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            with open(os.path.join(directory, file), "r") as f:
                data.append((file, json.load(f)))
    return data

def clean(df):
    def _clean(text):
        return text.replace("<<", "").replace(">>", "")
    
    for col in df.columns:
        if "text" in col:
            df[col] = df[col].apply(_clean)
    return df

def repeat_precision(df):
    grouped = df.groupby("id")
    results = []

    for name, group in grouped:
        if len(group) == 2:
            first_pass = group[(group['highlighted'] == False) & (group['ground_truth'] == True)]
            second_pass = group[(group['highlighted'] == True) & (group['ground_truth'] == True)]

            if not first_pass.empty and not second_pass.empty:
                first_marked_correct = first_pass.iloc[0]['predicted']
                second_marked_correct = second_pass.iloc[0]['predicted']
                quantile = group.iloc[0]['quantile']
                
                if first_marked_correct and second_marked_correct:
                    both_correct = "both"
                elif first_marked_correct:
                    both_correct = "first"
                elif second_marked_correct:
                    both_correct = "second"
                else:
                    both_correct = "neither"
                
                results.append({
                    'text': name,
                    'correct': both_correct,
                    'quantile': quantile
                })

    result_df = pd.DataFrame(results)
    quantile_grouped = result_df.groupby("quantile")
    per_quantile = {}

    for name, group in quantile_grouped:
        count = Counter(group['correct'])
        per_quantile[name] = {
            'first': count['first'],
            'second': count['second'],
            'both': count['both'],
            'neither': count['neither']
        }

    return per_quantile

def get_stats(data):
    df = pd.DataFrame(data)
    df = clean(df)
    per_quantile_repeat_precision = repeat_precision(df)
    return per_quantile_repeat_precision
# %%
# Create a figure for plotting
fig, axs = plt.subplots(3, 2, figsize=(15, 20))
axs = axs.flatten()

for i, layer_idx in enumerate(range(0, 12, 2)):
    ax = axs[i]
    
    for dir_idx, directory in enumerate(directories):
        data = load_data(directory)
        layer_stats = defaultdict(dict)

        for file, scores in data:
            layer = int(file.split("_")[0].split("layer")[-1])
            if layer == layer_idx:
                stats = get_stats(scores)
                for quantile, stat in stats.items():
                    if quantile not in layer_stats[layer]:
                        layer_stats[layer][quantile] = Counter(stat)
                    layer_stats[layer][quantile] += Counter(stat)

        # Extract 'both' data for the current layer
        both_data = {quantile: stat['both'] for quantile, stat in layer_stats[layer_idx].items()}
        
        # Sort the data by quantile
        sorted_data = sorted(both_data.items())
        quantiles, counts = zip(*sorted_data)
        
        # Plot the 'both' line for this directory
        if dir_idx+2 == 9:
            label = 'hand-written'
        else:
            label = f'nfew{dir_idx+2}'
        ax.plot(quantiles, counts, label=label, linewidth=1)

    ax.legend()
    ax.set_xlabel('Quantile')
    ax.set_ylabel('Count of "both" correct')
    ax.set_title(f'"Both" Correct Distribution (Layer {layer_idx})')
    ax.set_ylim(20, 90)

plt.tight_layout()
plt.show()
