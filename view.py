#%%

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        "logit_perplexity": []
    }
    
    for filename in os.listdir(directory):
        if filename.endswith(".json") and "layer8" in filename:
            with open(os.path.join(directory, filename), 'r') as file:
                json_data = json.load(file)
                data["activation_mean"].append(json_data["activation_mean"])
                data["activation_std"].append(json_data["activation_std"])
                data["activation_skew"].append(json_data["activation_skew"])
                data["activation_kurtosis"].append(json_data["activation_kurtosis"])
                data["activation_similarity"].append(json_data["activation_similarity"])
                data["logit_skew"].append(json_data["logit_skew"])
                data["logit_kurtosis"].append(json_data["logit_kurtosis"])
                data["logit_entropy"].append(json_data["logit_entropy"])
                data["logit_perplexity"].append(json_data["logit_perplexity"])

    return pd.DataFrame(data)

# Function to plot correlation matrix and KDE plots
def plot_correlations_kde(data):
    corr_matrix = data.corr()
    
    # fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    # axes = axes.flatten()
    
    # for i, key in enumerate(data.columns):
    #     sns.kdeplot(data[key], ax=axes[i], shade=True)
    #     axes[i].set_title(f'KDE Plot for {key}')
    #     axes[i].set_xlabel(key)
    #     axes[i].set_ylabel('Density')
    
    # plt.tight_layout()
    # plt.show()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

# Directory containing the JSON files
directory = 'new_processed_features'

# Read JSON files and extract statistics
data = read_json_files(directory)

# Plot KDE plots and correlation matrix
plot_correlations_kde(data)
# %%
