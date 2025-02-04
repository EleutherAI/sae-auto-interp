import torch
from torchtyping import TensorType

from .features import FeatureRecord, prepare_examples
from .loader import BufferOutput


def _top_k_pools(
        max_buffer: TensorType["batch"], 
        split_activations: TensorType["activations"], 
        buffer_tokens: TensorType["batch", "ctx_len"], 
        ctx_len: int, 
        max_examples: int
    ):
    """
    Get the top k activation pools.

    Args:
        max_buffer (TensorType["batch"]): The maximum buffer values.
        split_activations (List[TensorType["activations"]]): The split activations.
        buffer_tokens (TensorType["batch", "ctx_len"]): The buffer tokens.
        ctx_len (int): The context length.
        max_examples (int): The maximum number of examples.

    Returns:
        Tuple[TensorType["examples", "ctx_len"], TensorType["examples", "ctx_len"]]: The token windows and activation windows.
    """
    k = min(max_examples, len(max_buffer))
    top_values,top_indices = torch.topk(max_buffer, k, sorted=True)

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
    Pool max activation windows from the buffer output and update the feature record.

    Args:
        record (FeatureRecord): The feature record to update.
        buffer_output (BufferOutput): The buffer output containing activations and locations.
        tokens (TensorType["batch", "seq"]): The input tokens.
        ctx_len (int): The context length.
        max_examples (int): The maximum number of examples.
    """
    flat_indices = buffer_output.locations[:, 0] * tokens.shape[1] + buffer_output.locations[:, 1]
    ctx_indices = flat_indices//ctx_len
    index_within_ctx = flat_indices%ctx_len

    torch.testing.assert_close(ctx_indices, buffer_output.locations[:, 0])
    torch.testing.assert_close(index_within_ctx, buffer_output.locations[:, 1])

    # unique_ctx_indices: array of distinct context window indices in order of first appearance. i.e. sequential integers from 0 to 3903
    # inverses: maps each activation back to its index in unique_ctx_indices (can be used to dereference the context window idx of each activation)
    # lengths: the number of activations per unique context window index
    unique_ctx_indices, inverses, lengths = torch.unique_consecutive(ctx_indices, return_counts=True, return_inverse=True)

    # Get the max activation magnitude within each context window
    max_buffer = torch.segment_reduce(buffer_output.activations, 'max', lengths=lengths)

    # Create a zeros tensor with the same shape as the activations tensor
    new_tensor= torch.zeros(len(unique_ctx_indices), ctx_len, dtype=buffer_output.activations.dtype)

    # Deduplicate the context windows
    new_tensor[inverses, index_within_ctx] = buffer_output.activations

    # Does nothing if tokens is already at the context length
    buffer_tokens = tokens.reshape(-1, ctx_len)
    # Does nothing
    buffer_tokens = buffer_tokens[unique_ctx_indices]

    token_windows, activation_windows = _top_k_pools(
        max_buffer, new_tensor, buffer_tokens, ctx_len, max_examples
    )
    record.examples = prepare_examples(token_windows, activation_windows)

def random_activation_windows(
    record,
    tokens: TensorType["batch", "seq"],
    buffer_output: BufferOutput,
    ctx_len: int,
    n_random: int,
):
    """
    Generate random activation windows and update the feature record.

    Args:
        record (FeatureRecord): The feature record to update.
        tokens (TensorType["batch", "seq"]): The input tokens.
        buffer_output (BufferOutput): The buffer output containing activations and locations.
        ctx_len (int): The context length.
        n_random (int): The number of random examples to generate.
    """
    torch.manual_seed(22)
    batch_size = tokens.shape[0]
    unique_batch_pos = buffer_output.locations[:, 0].unique()

    mask = torch.ones(batch_size, dtype=torch.bool)
    mask[unique_batch_pos] = False

    available_indices = mask.nonzero().squeeze()
    selected_indices = available_indices[torch.randint(0,len(available_indices),size=(n_random,))]

    toks = tokens[selected_indices, 10 : 10 + ctx_len]

    record.random_examples = prepare_examples(
        toks,
        torch.zeros_like(toks),
    )

def default_constructor(
    record: FeatureRecord,
    tokens: TensorType["batch", "seq"],
    buffer_output: BufferOutput,
    n_random: int,
    ctx_len: int,
    max_examples: int,
):
    """
    Construct feature examples using pool max activation windows and random activation windows.

    Args:
        record (FeatureRecord): The feature record to update.
        tokens (TensorType["batch", "seq"]): The input tokens.
        buffer_output (BufferOutput): The buffer output containing activations and locations.
        n_random (int): Number of random examples to generate.
        ctx_len (int): Context length for each example.
        max_examples (int): Maximum number of examples to generate.
    """
    pool_max_activation_windows(
        record,
        tokens=tokens,
        buffer_output=buffer_output,
        ctx_len=ctx_len,
        max_examples=max_examples,
    )
    random_activation_windows(
        record,
        tokens=tokens,
        buffer_output=buffer_output,
        n_random=n_random,
        ctx_len=ctx_len,
    )
