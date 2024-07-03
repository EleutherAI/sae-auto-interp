#%%

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
        if filename.endswith(".json") and "layer10" in filename:
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
                data["unique_tokens"].append(json_data["unique_tokens"])

    print(f"Number of JSON files read: {len(data['activation_mean'])}")

    return pd.DataFrame(data)

# Function to plot correlation matrix and KDE plots
def plot_correlations_kde(data):
    corr_matrix = data.corr()

    # Mask for the upper triangle
    mask = np.triu(corr_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, mask=mask, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix (Lower Triangular)')
    plt.show()

# Directory containing the JSON files
directory = '/share/u/caden/sae-auto-interp/new_processed_features'

# Read JSON files and extract statistics
data = read_json_files(directory)

# Plot KDE plots and correlation matrix
plot_correlations_kde(data)
# %%


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# %%
import torch
tokens =[ torch.tensor([0,34,34,1,4,1])]

tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
