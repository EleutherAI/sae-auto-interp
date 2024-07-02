
from sentence_transformers import SentenceTransformer
import torch
import matplotlib.pyplot as plt

# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_average_similarity(sentences):
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cos_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    n = cos_sim.size(0)
    average_similarity = (cos_sim.sum() - cos_sim.trace()) / (n * (n - 1))
    return average_similarity.item()

n_values = range(100, 1001, 50)  # From 10 to 100, step 10
avg_similarities = []

for record_index in range(20):
    r = layer_record_dict[0][record_index]
    sentences = [s.text for s in r.examples]
    
    record_similarities = []
    for n in tqdm(n_values, desc="Calculating average similarity"):
        avg_sim = calculate_average_similarity(sentences[:n])
        record_similarities.append(avg_sim)
    
    avg_similarities.append(record_similarities)

# Calculate the average similarity across all records for each n
avg_similarities_across_records = [sum(sims) / len(sims) for sims in zip(*avg_similarities)]

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(n_values, avg_similarities_across_records, marker='o')
plt.xlabel('Number of sentences (n)')
plt.ylabel('Average similarity')
plt.title('Average similarity vs. Number of sentences')
plt.grid(True)
plt.show()

# %%

import matplotlib.pyplot as plt
import numpy as np

def normalize_log_activations(log_acts):
    min_val = np.min(log_acts)
    max_val = np.max(log_acts)
    return (log_acts - min_val) / (max_val - min_val)

for layer, records in layer_record_dict.items():
    filtered_records = [
        record for record in records if len(record.n_activations) > 600
    ]

    n_records = len(filtered_records)
    
    fig, axs = plt.subplots(len(filtered_records), figsize=(8, 3*n_records))

    for i, record in enumerate(filtered_records):
        acts = np.array(record.top_activations)
        log_acts = np.log(acts)
        normalized_log_acts = normalize_log_activations(log_acts)
        
        # Create histogram with normalized axes
        axs[i].hist(normalized_log_acts, bins=50, edgecolor='black', alpha=0.7, density=True)

        # Add text annotations
        axs[i].text(0.05, 0.90, f'n: {len(record.n_activations)}', transform=axs[i].transAxes, 
                verticalalignment='top', color='red')
        axs[i].text(0.05, 0.70, f'n_lemmas: {len(record.lemmas)}', transform=axs[i].transAxes, 
                       verticalalignment='top', color='green')
        axs[i].text(0.05, 0.50, f'n_acts: {np.average(record.n_activations):.2f}', transform=axs[i].transAxes, 
                       verticalalignment='top', color='blue')

        feature_index = record.feature.feature_index
        layer_index = record.feature.layer_index
        axs[i].set_title(f'{layer_index}_{feature_index}')
        axs[i].set_xlabel('Normalized Log(Activations)')
        axs[i].set_ylabel('Density')

        # Set consistent x and y limits
        axs[i].set_xlim(0, 1)
        axs[i].set_ylim(0, 5)  # You may need to adjust this value

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'normalized_histogram_plot_layer{layer}.png')