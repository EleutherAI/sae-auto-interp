from collections import defaultdict
from math import floor

import numpy as np
import torch
import torch.nn.functional as F

from . import LatentRecord


def logits(
    records: list[LatentRecord],
    W_U: torch.nn.Module,
    W_dec: torch.nn.Module,
    k: int = 10,
    tokenizer=None,
) -> list[list[str]]:
    """
    Compute the top k logits via direct logit attribution for a set of records.

    Args:
        records (list[LatentRecord]): A list of latent records.
        W_U (torch.nn.Module): The linear layer for the encoder.
        W_dec (torch.nn.Module): The linear layer for the decoder.
        k (int): The number of top logits to compute.
        tokenizer (Optional): A tokenizer for decoding logits.

    Returns:
        decoded_top_logits (list[list[str]]): A list of top k logits for each record.
    """

    latent_indices = [record.latent.latent_index for record in records]

    narrowed_logits = torch.matmul(W_U, W_dec[:, latent_indices])

    top_logits = torch.topk(narrowed_logits, k, dim=0).indices

    per_example_top_logits = top_logits.T

    decoded_top_logits = []

    for record_index in range(len(records)):
        decoded = tokenizer.batch_decode(per_example_top_logits[record_index])
        decoded_top_logits.append(decoded)

        records[record_index].top_logits = decoded


def unigram(
    record: LatentRecord, k: int = 10, threshold: float = 0.0, negative_shift: int = 0
):
    avg_nonzero = []
    top_tokens = []

    n_examples = floor(len(record.examples) * threshold)

    for example in record.examples[:n_examples]:
        # Get the number of nonzero activations per example
        avg_nonzero.append(np.count_nonzero(example.activations))

        # Get the max activating token per example
        index = np.argmax(example.activations) - negative_shift

        if index < 0:
            continue

        top_tokens.append(example.tokens[index].item())

    if len(set(top_tokens)) < k:
        return set(top_tokens), np.mean(avg_nonzero)

    return -1, np.mean(avg_nonzero)


def cos(matrix, selected_latents=[0]):
    a = matrix[:, selected_latents]
    b = matrix

    a = F.normalize(a, p=2, dim=0)
    b = F.normalize(b, p=2, dim=0)

    cos_sim = torch.mm(a.t(), b)

    return cos_sim


def get_neighbors(submodule_dict, latent_filter, k=10):
    """
    Get the required latents for neighbor scoring.

    Returns:
        neighbors_dict: Nested dictionary of modules -> neighbors -> indices, values
        per_layer_latents (dict): A dictionary of latents per layer
    """

    neighbors_dict = defaultdict(dict)
    per_layer_latents = {}

    for module_path, submodule in submodule_dict.items():
        selected_latents = latent_filter.get(module_path, False)
        if not selected_latents:
            continue

        W_D = submodule.ae.autoencoder._module.decoder.weight
        cos_sim = cos(W_D, selected_latents=selected_latents)
        top = torch.topk(cos_sim, k=k)

        top_indices = top.indices
        top_values = top.values

        for i, (indices, values) in enumerate(zip(top_indices, top_values)):
            neighbors_dict[module_path][i] = {
                "indices": indices.tolist()[1:],
                "values": values.tolist()[1:],
            }

        per_layer_latents[module_path] = torch.unique(top_indices).tolist()

    return neighbors_dict, per_layer_latents
