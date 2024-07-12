# %%

import torch
def range_indices(start_positions, N):
    range_tensor = torch.arange(N)

    range_broadcasted = range_tensor.unsqueeze(0).expand(
        start_positions.size(0), -1
    )

    indices = start_positions.unsqueeze(1) + range_broadcasted

    return indices

def find_max_activation_slices(locations, activations, tokens, N, k=10):
    batch_len, seq_len = tokens.shape
    
    # Create a sparse tensor of activations
    sparse_activations = torch.sparse_coo_tensor(locations.t(), activations, (batch_len, seq_len))
    dense_activations = sparse_activations.to_dense()
    
    # Compute cumulative sum of activations along sequence dimension
    cumsum_activations = torch.cumsum(dense_activations, dim=1)
    
    # Compute sliding window sums of length N
    window_sums = cumsum_activations[:, N:] - cumsum_activations[:, :-N]

    # Find the maximum sum and its location
    _, top_k_indices = torch.topk(window_sums.flatten(), k)

    # Convert flat indices to batch and sequence indices
    batch_indices = top_k_indices // (seq_len - N)
    seq_indices = top_k_indices % (seq_len - N)

    batch_indices = batch_indices.unsqueeze(1).expand(-1, N)
    seq_indices = range_indices(seq_indices, N)

    max_slice_tokens = tokens[batch_indices, seq_indices]
    max_slice_activations = dense_activations[batch_indices, seq_indices]
    return max_slice_tokens, max_slice_activations

# %%

raw_dir = "raw_features"

# Build location paths
locations_path = f"{raw_dir}/layer{0}_locations.pt"
activations_path = f"{raw_dir}/layer{0}_activations.pt"

# Load tensor
locations = torch.load(locations_path)
activations = torch.load(activations_path)

mask = locations[:, 2] == 0
feature_locations = locations[mask]
feature_activations = activations[mask]

# %%

import os
os.environ["CONFIG_PATH"] = "configs/caden_gpt2.yaml"
from sae_auto_interp.utils import load_tokenized_data
from nnsight import LanguageModel

# tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
# tokenizer.padding_side = "left"

tokens = load_tokenized_data(model.tokenizer)


# %%

def pool_max_activation_slices(
    locations, activations, tokens, N, k=10
):
    batch_len, seq_len = tokens.shape

    sparse_activations = torch.sparse_coo_tensor(locations[:,:2].t(), activations, (batch_len, seq_len))
    dense_activations = sparse_activations.to_dense()


    unique_batch_pos = torch.unique(locations[:,0])
    token_batches = tokens[unique_batch_pos]
    dense_activations = dense_activations[unique_batch_pos]

    avg_pools = torch.nn.functional.max_pool1d(
        dense_activations, kernel_size=N, stride=N
    )

    activation_windows = dense_activations.unfold(1, N, N).reshape(-1, N)
    token_windows = token_batches.unfold(1, N, N).reshape(-1, N)

    top_indices = torch.topk(avg_pools.flatten(), k).indices

    activation_windows = activation_windows[top_indices]
    token_windows = token_windows[top_indices]

    return token_windows, activation_windows


# %%

def get_activating_examples(
    tokens: torch.Tensor, 
    locations: torch.Tensor, 
    activations: torch.Tensor
):
    # Create a sparse tensor for activations
    num_sentences, sentence_length = tokens.shape
    indices = torch.stack((
        locations[:, 0].long(), 
        locations[:, 1].long()
    ))
    
    sparse_activations = torch.sparse_coo_tensor(
        indices, 
        activations, 
        (num_sentences, sentence_length)
    )
    
    # Convert to dense and find active sentences
    dense_activations = sparse_activations.to_dense()
    active_sentences = torch.any(dense_activations != 0, dim=1)
    
    return tokens[active_sentences], dense_activations[active_sentences]

def extract_activation_windows(tokens: torch.Tensor, activations: torch.Tensor, l_ctx: int = 15, r_ctx: int = 4):
    """
    Extracts windows around the maximum activation for each sentence.
    
    Args:
        tokens: Tensor of tokens for activating sentences
        activations: Tensor of activations for activating sentences
        l_ctx: Number of tokens to the left
        r_ctx: Number of tokens to the right
    
    Returns:
        List of token windows and activation windows
    """
    # Find the token with max activation for each sentence
    max_activation_indices = torch.argmax(activations, dim=1)

    # Calculate start and end indices for the windows
    start_indices = torch.clamp(max_activation_indices - l_ctx, min=0)
    end_indices = torch.clamp(max_activation_indices + r_ctx + 1, max=tokens.shape[1])

    # Initialize lists to store results
    token_windows = []
    activation_windows = []

    # Extract windows (this part is hard to vectorize due to variable window sizes)
    for i, (start, end) in enumerate(zip(start_indices, end_indices)):
        token_windows.append(tokens[i, start:end])
        activation_windows.append(activations[i, start:end])

    return token_windows, activation_windows


# %%

import timeit 
def get_non_activating_tokens(
    locations, tokens, n_to_find, ctx_len=20
):
    unique_batch_pos = torch.unique(locations[:,0])
    taken = set(unique_batch_pos.tolist())
    free = []
    value = 0
    while value < n_to_find:
        if value not in taken:
            free.append(value)
            value += 1
    return tokens[free, 10:10+ctx_len]

start_time = timeit.default_timer()
val = get_non_activating_tokens(feature_locations[:, :2], tokens, 20)
print(timeit.default_timer() - start_time)

val

# %%

import numpy as np
import matplotlib.pyplot as plt
import timeit
from tqdm import tqdm

# Function to measure time for pool_avg_activation_slices
def time_pool_avg(k, runs=5):
    times = []
    for _ in range(runs):
        start_time = timeit.default_timer()
        pool_max_activation_slices(feature_locations[:,:2], feature_activations, tokens, N=20, k=k)
        times.append(timeit.default_timer() - start_time)
    return np.mean(times)

# Function to measure time for find_max_activation_slices
def time_find_max(k, runs=5):
    times = []
    for _ in range(runs):
        start_time = timeit.default_timer()
        find_max_activation_slices(feature_locations[:,:2], feature_activations, tokens, N=20, k=k)
        times.append(timeit.default_timer() - start_time)
    return np.mean(times)


def time_find_windows(k, runs=5):
    times = []
    for _ in range(runs):
        start_time = timeit.default_timer()
        example_tokens, example_activations = get_activating_examples(
            tokens, locations, activations
        )

        extract_activation_windows(
            example_tokens[:k], 
            example_activations[:k]
        )
        times.append(timeit.default_timer() - start_time)
    return np.mean(times)

ks = range(100, 2000, 100)
pool_avg_times = []
find_max_times = []
find_windows_times = []

for k in tqdm(ks, desc="Timing functions"):
    pool_avg_times.append(time_pool_avg(k))
    find_max_times.append(time_find_max(k))
    find_windows_times.append(time_find_windows(k))


plt.figure(figsize=(10, 6))
plt.plot(ks, pool_avg_times, label='pool_avg_activation_slices')
plt.plot(ks, find_max_times, label='find_max_activation_slices')
plt.plot(ks, find_windows_times, label='find_activation_windows')
plt.xlabel('k')
plt.ylabel('Time (seconds)')
plt.title('Function Timings for Increasing Values of k')
plt.legend()
plt.grid(True)
plt.show()

