import torch
from torchtyping import TensorType  
from .features import Example


def pool_max_activation_windows(
    record,
    tokens: TensorType["batch", "seq"],
    locations: TensorType["locations", 2],
    activations: TensorType["locations"],  
    k: int = 10,
    ctx_len: int = 20
):
    batch_len, seq_len = tokens.shape

    sparse_activations = torch.sparse_coo_tensor(
        locations.t(), activations, (batch_len, seq_len)
    )
    dense_activations = sparse_activations.to_dense()

    unique_batch_pos = torch.unique(locations[:,0])
    token_batches = tokens[unique_batch_pos]
    dense_activations = dense_activations[unique_batch_pos]

    avg_pools = torch.nn.functional.max_pool1d(
        dense_activations, kernel_size=ctx_len, stride=ctx_len
    )

    activation_windows = dense_activations.unfold(1, ctx_len, ctx_len).reshape(-1, ctx_len)
    token_windows = token_batches.unfold(1, ctx_len, ctx_len).reshape(-1, ctx_len)

    # Should add this back in?
    # non_zero = avg_pools != 0
    # non_zero = non_zero.sum()
    k = min(k, len(avg_pools))
    
    top_indices = torch.topk(avg_pools.flatten(), k).indices

    activation_windows = activation_windows[top_indices]
    token_windows = token_windows[top_indices]

    record.examples = Example.prepare_examples(
        token_windows, activation_windows
    )


def random_activation_windows(
    record,
    tokens: torch.Tensor,
    locations: torch.Tensor,
    n_to_find: int = 10, 
    ctx_len: int = 20
):
    torch.manual_seed(22)
    batch_size = tokens.shape[0]
    unique_batch_pos = locations[:, 0].unique()
    
    mask = torch.ones(batch_size, dtype=torch.bool)
    mask[unique_batch_pos] = False
    
    available_indices = mask.nonzero().squeeze()
    
    selected_indices = available_indices[
        torch.randperm(len(available_indices))[:n_to_find]
    ]
    
    toks = tokens[selected_indices, 10:10+ctx_len]
    
    record.random_examples = Example.prepare_examples(
        toks, torch.zeros_like(toks),
    )
