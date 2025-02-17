from typing import Callable, Literal, Optional

import torch
from torchtyping import TensorType

from .latents import LatentRecord, prepare_examples
from .loader import ActivationData


def _top_k_pools(
    max_buffer: TensorType["batch"],
    split_activations: list[TensorType["activations"]],
    buffer_tokens: TensorType["batch", "ctx_len"],
    max_examples: int,
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
        Tuple[TensorType["examples", "ctx_len"], TensorType["examples", "ctx_len"]]:
        The token windows and activation windows.
    """
    k = min(max_examples, len(max_buffer))
    top_values, top_indices = torch.topk(max_buffer, k, sorted=True)

    activation_windows = torch.stack([split_activations[i] for i in top_indices])
    token_windows = buffer_tokens[top_indices]

    return token_windows, activation_windows


def pool_max_activation_windows(
    activations: TensorType["n_examples"],
    tokens: TensorType["windows", "seq"],
    ctx_indices: TensorType["n_examples"],
    index_within_ctx: TensorType["n_examples"],
    ctx_len: int,
    max_examples: int,
):
    """
    Pool max activation windows from the buffer output and update the latent record.

    Args:
        activations (TensorType["n_examples"]): The activations.
        tokens (TensorType["windows", "seq"]): The input tokens.
        ctx_indices (TensorType["n_examples"]): The context indices.
        index_within_ctx (TensorType["n_examples"]): The index within the context.
        ctx_len (int): The context length.
        max_examples (int): The maximum number of examples.
    """
    # unique_ctx_indices: array of distinct context window indices in order of first
    # appearance. sequential integers from 0 to batch_size * cache_token_length//ctx_len
    # inverses: maps each activation back to its index in unique_ctx_indices
    # (can be used to dereference the context window idx of each activation)
    # lengths: the number of activations per unique context window index
    unique_ctx_indices, inverses, lengths = torch.unique_consecutive(
        ctx_indices, return_counts=True, return_inverse=True
    )

    # Get the max activation magnitude within each context window
    max_buffer = torch.segment_reduce(activations, "max", lengths=lengths)

    # Deduplicate the context windows
    new_tensor = torch.zeros(len(unique_ctx_indices), ctx_len, dtype=activations.dtype)
    new_tensor[inverses, index_within_ctx] = activations

    tokens = tokens[unique_ctx_indices]

    token_windows, activation_windows = _top_k_pools(
        max_buffer, new_tensor, tokens, max_examples
    )

    return token_windows, activation_windows


def constructor(
    record: LatentRecord,
    activation_data: ActivationData,
    n_not_active: int,
    max_examples: int,
    ctx_len: int,
    constructor_type: Literal["random", "neighbour"],
    token_loader: Optional[Callable[[], TensorType["batch", "seq"]]] | None = None,
    all_data: Optional[ActivationData] = None,
):
    tokens = activation_data.tokens
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

    tokens.shape[0]
    cache_token_length = tokens.shape[1]

    # Get all positions where the latent is active
    flat_indices = (
        activation_data.locations[:, 0] * cache_token_length
        + activation_data.locations[:, 1]
    )
    ctx_indices = flat_indices // ctx_len
    index_within_ctx = flat_indices % ctx_len
    reshaped_tokens = tokens.reshape(-1, ctx_len)
    n_windows = reshaped_tokens.shape[0]

    unique_batch_pos = ctx_indices.unique()

    mask = torch.ones(n_windows, dtype=torch.bool)
    mask[unique_batch_pos] = False
    # Indices where the latent is active
    active_indices = mask.nonzero(as_tuple=False).squeeze()
    activations = activation_data.activations

    # Add activation examples to the record in place
    token_windows, act_windows = pool_max_activation_windows(
        activations=activations,
        tokens=reshaped_tokens,
        ctx_indices=ctx_indices,
        index_within_ctx=index_within_ctx,
        ctx_len=ctx_len,
        max_examples=max_examples,
    )
    record.examples = prepare_examples(token_windows, act_windows)

    if constructor_type == "random":
        # Add random non-activating examples to the record in place
        random_non_activating_windows(
            record,
            available_indices=active_indices,
            reshaped_tokens=reshaped_tokens,
            n_not_active=n_not_active,
        )
    elif constructor_type == "neighbour":
        neighbour_non_activation_windows(
            record,
            not_active_mask=mask,
            tokens=tokens,
            all_data=all_data,
            ctx_len=ctx_len,
            n_not_active=n_not_active,
        )


def neighbour_non_activation_windows(
    record: LatentRecord,
    not_active_mask: TensorType["n_windows"],
    tokens: TensorType["batch", "seq"],
    all_data: ActivationData,
    ctx_len: int,
    n_not_active: int,
):
    """
    Generate random activation windows and update the latent record.

    Args:
        record (LatentRecord): The latent record to update.
        not_active_mask (TensorType["n_windows"]): The mask of the non-active windows.
        tokens (TensorType["batch", "seq"]): The input tokens.
        all_data (AllData): The all data containing activations and locations.
        ctx_len (int): The context length.
        n_random (int): The number of random examples to generate.
    """
    torch.manual_seed(22)
    if n_not_active == 0:
        record.not_active = []
        return

    assert (
        record.neighbours is not None
    ), "Neighbours are not set, add them via a transform"

    cache_token_length = tokens.shape[1]
    reshaped_tokens = tokens.reshape(-1, ctx_len)
    n_windows = reshaped_tokens.shape[0]
    # TODO: For now we use at most 10 examples per neighbour, we may want to allow a
    # variable number of examples per neighbour
    n_examples_per_neighbour = 10

    number_examples = 0
    available_features = all_data.features
    all_examples = []
    used_neighbours = []
    for neighbour in record.neighbours:
        if number_examples >= n_not_active:
            break
        # find indice in all_data.features that matches the neighbour
        indice = torch.where(available_features == neighbour.feature_index)[0]
        if len(indice) == 0:
            continue
        # get the locations of the neighbour
        locations = all_data.locations[indice]
        activations = all_data.activations[indice]
        # get the active window indices
        flat_indices = locations[:, 0] * cache_token_length + locations[:, 1]
        ctx_indices = flat_indices // ctx_len
        index_within_ctx = flat_indices % ctx_len
        # Set the mask to True for the unique locations
        unique_batch_pos_active = ctx_indices.unique()

        mask = torch.zeros(n_windows, dtype=torch.bool)
        mask[unique_batch_pos_active] = True

        # Get the indices where mask and not_active_mask are True
        mask = mask & not_active_mask

        available_indices = mask.nonzero().flatten()

        mask_ctx = torch.isin(ctx_indices, available_indices)
        available_ctx_indices = ctx_indices[mask_ctx]
        available_index_within_ctx = index_within_ctx[mask_ctx]
        activations = activations[mask_ctx]
        # If there are no available indices, skip this neighbour
        if activations.numel() == 0:
            continue
        token_windows, act_windows = pool_max_activation_windows(
            activations=activations,
            tokens=reshaped_tokens,
            ctx_indices=available_ctx_indices,
            index_within_ctx=available_index_within_ctx,
            max_examples=n_examples_per_neighbour,
            ctx_len=ctx_len,
        )
        # use the first n_examples_per_neighbour examples,
        # which will be the most active examples
        examples_used = len(token_windows)
        all_examples.append(
            prepare_examples(token_windows, torch.zeros_like(token_windows))
        )
        used_neighbours.append(neighbour)
        number_examples += examples_used
    record.neighbours = used_neighbours
    if len(all_examples) == 0:
        print("No examples found")

    record.not_active = all_examples


def random_non_activating_windows(
    record: LatentRecord,
    available_indices: TensorType["n_windows"],
    reshaped_tokens: TensorType["n_windows", "ctx_len"],
    n_not_active: int,
):
    """
    Generate random non-activating sequence windows and update the latent record.

    Args:
        record (LatentRecord): The latent record to update.
        available_indices (TensorType["n_windows"]): The indices of the windows where
        the latent is not active.
        reshaped_tokens (TensorType["n_windows", "ctx_len"]): The tokens reshaped
        to the context length.
        n_not_active (int): The number of non activating examples to generate.
    """
    torch.manual_seed(22)
    if n_not_active == 0:
        record.not_active = []
        return

    # If this happens it means that the latent is active in every window,
    # so it is a bad latent
    if available_indices.numel() < n_not_active:
        print("No available randomly sampled non-activating sequences")
        record.not_active = []
        return
    else:
        random_indices = torch.randint(
            0, available_indices.shape[0], size=(n_not_active,)
        )
        selected_indices = available_indices[random_indices]

    toks = reshaped_tokens[selected_indices]

    record.not_active = prepare_examples(
        toks,
        torch.zeros_like(toks),
    )
