from typing import Callable, Optional

import torch
from torchtyping import TensorType
from .latents import LatentRecord, prepare_examples
from .loader import BufferOutput, AllData

def _top_k_pools(
        max_buffer: TensorType["batch"],
        split_activations: list[TensorType["activations"]], 
        buffer_tokens: TensorType["batch", "ctx_len"], 
        max_examples: int
    ):
    """
    Get the top k activation pools.

    Args:
        max_buffer (TensorType["batch"]): The maximum buffer values.
        split_activations (list[TensorType["activations"]]): The split activations.
        buffer_tokens (TensorType["batch", "ctx_len"]): The buffer tokens.
        ctx_len (int): The context length.
        max_examples (int): The maximum number of examples.

    Returns:
        Tuple[TensorType["examples", "ctx_len"], TensorType["examples", "ctx_len"]]: The token windows and activation windows.
    """
    k = min(max_examples, len(max_buffer))
    top_values, top_indices = torch.topk(max_buffer, k, sorted=True)

    activation_windows = torch.stack([split_activations[i] for i in top_indices])
    token_windows = buffer_tokens[top_indices]

    return token_windows, activation_windows

def pool_max_activation_windows(
    record,
    buffer_output: BufferOutput,
    tokens: TensorType["batch", "seq"],
    ctx_len: int,
    max_examples: int,
):
    """
    Pool max activation windows from the buffer output and update the latent record.

    Args:
        record (LatentRecord): The latent record to update.
        buffer_output (BufferOutput): The buffer output containing activations and locations.
        tokens (TensorType["batch", "seq"]): The input tokens.
        ctx_len (int): The context length.
        max_examples (int): The maximum number of examples.
    """
    flat_indices = buffer_output.locations[:, 0] * tokens.shape[1] + buffer_output.locations[:, 1]
    ctx_indices = flat_indices // ctx_len
    index_within_ctx = flat_indices % ctx_len
  
    # unique_ctx_indices: array of distinct context window indices in order of first appearance. i.e. sequential integers from 0 to 3903
    # inverses: maps each activation back to its index in unique_ctx_indices (can be used to dereference the context window idx of each activation)
    # lengths: the number of activations per unique context window index
    unique_ctx_indices, inverses, lengths = torch.unique_consecutive(ctx_indices, return_counts=True, return_inverse=True)
    # Get the max activation magnitude within each context window
    max_buffer = torch.segment_reduce(buffer_output.activations, 'max', lengths=lengths)

    # Deduplicate the context windows
    new_tensor= torch.zeros(len(unique_ctx_indices), ctx_len, dtype=buffer_output.activations.dtype)
    new_tensor[inverses, index_within_ctx] = buffer_output.activations

    buffer_tokens = tokens.reshape(-1, ctx_len)
    buffer_tokens = buffer_tokens[unique_ctx_indices]

    token_windows, activation_windows = _top_k_pools(
        max_buffer, new_tensor, buffer_tokens, max_examples
    )

    record.examples = prepare_examples(token_windows, activation_windows)

def random_non_activating_windows(
    record: LatentRecord,
    tokens: TensorType["batch", "seq"],
    buffer_output: BufferOutput,
    ctx_len: int,
    n_not_active: int,
):
    """
    Generate random non-activating sequence windows and update the latent record.

    Args:
        record (LatentRecord): The latent record to update.
        tokens (TensorType["batch", "seq"]): The input tokens.
        buffer_output (BufferOutput): The buffer output containing activations and locations.
        ctx_len (int): The context length.
        n_not_active (int): The number of non activating examples to generate.
    """
    torch.manual_seed(22)
    if n_not_active == 0:
        record.not_active = []
        return
    
    batch_size = tokens.shape[0]
    unique_batch_pos = buffer_output.locations[:, 0].unique()

    mask = torch.ones(batch_size, dtype=torch.bool)
    mask[unique_batch_pos] = False

    available_indices = mask.nonzero().squeeze()

    # TODO:What to do when the latent is active at least once in each batch?
    if available_indices.numel() < n_not_active:
        print("No available randomly sampled non-activating sequences")
        record.not_active = []
        return
    else:
        selected_indices = available_indices[torch.randint(0,len(available_indices),size=(n_not_active,))]

    toks = tokens[selected_indices, 10 : 10 + ctx_len]

    record.not_active = prepare_examples(
        toks,
        torch.zeros_like(toks),
    )

def neighbour_random_activation_windows(
    record: LatentRecord,
    tokens: TensorType["batch", "seq"],
    buffer_output: BufferOutput,
    all_data: AllData,
    ctx_len: int,
    n_random: int,
):
    """
    Generate random activation windows and update the latent record.

    Args:
        record (LatentRecord): The latent record to update.
        tokens (TensorType["batch", "seq"]): The input tokens.
        buffer_output (BufferOutput): The buffer output containing activations and locations.
        ctx_len (int): The context length.
        n_random (int): The number of random examples to generate.
    """
    torch.manual_seed(22)
    if n_random == 0:
        return
    
    assert record.neighbours is not None, "Neighbours are not set, add them via a transform"

    batch_size = tokens.shape[0]
    
    # Get the unique batch positions where the latent is active
    unique_batch_pos_active = buffer_output.locations[:, 0].unique_consecutive()
    
    mask = torch.zeros(batch_size, dtype=torch.bool)

    # TODO: For now we use at most 10 examples per neighbour, we may want to allow a variable number of examples per neighbour
    n_examples_per_neighbour = 10

    number_examples = 0
    available_features = all_data.features
    all_examples = []
    for neighbour in record.neighbours:
        if number_examples >= n_random:
            break
        # find indice in all_data.features that matches the neighbour
        indice = torch.where(available_features == neighbour.feature_index)[0]
        if len(indice) == 0:
            continue
        # get the locations of the neighbour
        locations = all_data.locations[indice]
        # get the unique locations
        unique_locations = locations[:,0].unique_consecutive(dim=0)

        # Set the mask to True for the unique locations
        mask[unique_locations] = True

        # Set the mask to False for the unique locations where the latent is active
        # TODO: we probably want to be less strict here, we could use parts of the batch where the latent is not active
        mask[unique_batch_pos_active] = False

        available_indices = mask.nonzero().flatten()

        if available_indices.numel() == 0:
            continue
        size = min(n_examples_per_neighbour, len(available_indices))
    
        selected_indices = torch.randint(0, len(unique_locations), size=(size,))
        selected_positions = torch.randint(0, tokens.shape[1] - ctx_len, size=(size,))
    
        range_indices = torch.arange(ctx_len, device=tokens.device).unsqueeze(0)  # Shape: (1, ctx_len)
    
        # Each selected_positions gives a unique starting index. We add the range tensor to get indices for each example.
        positions = selected_positions.unsqueeze(1) + range_indices    

        # Get tokens
        toks = tokens[selected_indices].gather(dim=1, index=positions)

        examples = prepare_examples(toks, torch.zeros_like(toks))
        number_examples += len(examples)
        all_examples.append((examples, neighbour))

    if len(all_examples) == 0:
        print("No examples found")

    record.random_examples = all_examples


def default_constructor(
    record: LatentRecord,
    token_loader: Optional[Callable[[], TensorType["batch", "seq"]]] | None,
    buffer_output: BufferOutput,
    n_not_active: int,
    ctx_len: int,
    max_examples: int,
):
    """
    Construct latent examples using pool max activation windows and random activation windows.

    Args:
        record (LatentRecord): The latent record to update.
        token_loader (Optional[Callable[[], TensorType["batch", "seq"]]]):
            An optional function that creates the dataset tokens.
        buffer_output (BufferOutput): The buffer output containing activations and locations.
        n_not_active (int): Number of non-activating examples to randomly generate.
        ctx_len (int): Context length for each example.
        max_examples (int): Maximum number of examples to generate.
    """
    tokens = buffer_output.tokens
    if tokens is None:
        if token_loader is None:
            raise ValueError("Either tokens or token_loader must be provided")
        try:
            tokens = token_loader()
        except TypeError:
            raise ValueError(
                "Starting with v0.2, `tokens` was renamed to `token_loader`, "
                "which must be a callable for lazy loading.\n\n"
                "Instead of passing\n"
                "`    tokens=dataset.tokens`,\n"
                "pass\n"
                "`    token_loader=lambda: dataset.load_tokens()`,\n"
                "(assuming `dataset` is a `LatentDataset` instance)."
            )
    pool_max_activation_windows(
        record,
        buffer_output=buffer_output,
        tokens=tokens,
        ctx_len=ctx_len,
        max_examples=max_examples,
    )
    random_non_activating_windows(
        record,
        tokens=tokens,
        buffer_output=buffer_output,
        n_not_active=n_not_active,
        ctx_len=ctx_len,
    )

def neighbour_constructor(
    record: LatentRecord,
    token_loader: Optional[Callable[[], TensorType["batch", "seq"]]],
    buffer_output: BufferOutput,
    all_data: AllData,
    n_random: int,
    ctx_len: int,
    max_examples: int,
):
    """
    Construct feature examples using pool max activation windows and random activation windows from neighbours.
    """
    tokens = all_data.tokens
    if tokens is None:
        if token_loader is None:
            raise ValueError("Either tokens or token_loader must be provided")
        try:
            tokens = token_loader()
        except TypeError:
            raise ValueError(
                "Starting with v0.2, `tokens` was renamed to `token_loader`, "
                "which must be a callable for lazy loading.\n\n"
                "Instead of passing\n"
                "`    tokens=dataset.tokens`,\n"
                "pass\n"
                "`    token_loader=lambda: dataset.load_tokens()`,\n"
                "(assuming `dataset` is a `FeatureDataset` instance)."
            )
    pool_max_activation_windows(
        record,
        buffer_output=buffer_output,
        tokens=tokens,
        ctx_len=ctx_len,
        max_examples=max_examples,
    )
    neighbour_random_activation_windows(
        record,
        tokens=tokens,
        buffer_output=buffer_output,
        all_data=all_data,
        n_random=n_random,
        ctx_len=ctx_len,
    )
    