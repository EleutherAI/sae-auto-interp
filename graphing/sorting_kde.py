# %%

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


for layer in range(0,12,2):


    # Function to read JSON files and extract the relevant statistics
    def read_json_files(directory):
        data = {
            "activation_mean": [],
            "activation_std": [],
            "activation_skew": [],
            "activation_kurtosis": [],
            "activation_similarity": [],
            "logit_skew": [],
            "logit_kurtosis": [],
            "logit_entropy": [],
            "logit_perplexity": [],
            "unique_tokens": []
        }
        
        for filename in os.listdir(directory):
            if filename.endswith(".json") and f"layer{layer}" in filename:
                with open(os.path.join(directory, filename), 'r') as file:
                    json_data = json.load(file)
                    for key in data.keys():
                        data[key].append(json_data[key])

        return pd.DataFrame(data)

    # Function to plot KDE correlation plots with normalized axes
    def plot_kde_correlations_normalized(data):
        measures = [col for col in data.columns if col != 'unique_tokens']
        n_measures = len(measures)
        fig, axes = plt.subplots(3, 3, figsize=(20, 20))
        axes = axes.flatten()

        # Normalize the data
        scaler = MinMaxScaler()
        data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        for i, measure in enumerate(measures):
            sns.kdeplot(
                data=data_normalized,
                x='unique_tokens',
                y=measure,
                ax=axes[i],
                cmap="YlGnBu",
                shade=True,
                cbar=True
            )
            axes[i].set_title(f'Unique Tokens vs {measure.replace("_", " ").title()}')
            axes[i].set_xlabel('Unique Tokens (Normalized)')
            axes[i].set_ylabel(f'{measure.replace("_", " ").title()} (Normalized)')
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)

        # Remove any unused subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    # Directory containing the JSON files
    directory = '/share/u/caden/sae-auto-interp/new_processed_features'

    # Read JSON files and extract statistics
    data = read_json_files(directory)

    # Plot KDE correlation plots with normalized axes
    plot_kde_correlations_normalized(data)
# %%
