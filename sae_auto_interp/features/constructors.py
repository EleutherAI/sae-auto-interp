import torch
from torchtyping import TensorType  

class Constructor:

    def __call__(self, tokens, **kwargs):
        self.construct(tokens, **kwargs)


class ActivationWindowConstructor(Constructor):
    def construct(
        self, 
        tokens: TensorType["batch", "seq"],
        locations: TensorType["locations", 2] = None,
        activations: TensorType["locations"] = None,  
        ctx_len: int = 20, 
        k: int = 10
    ):
        assert locations is not None
        assert activations is not None

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

        return token_windows, activation_windows


class RandomWindowConstructor(Constructor):
    def construct(
        self, 
        tokens: TensorType["batch", "seq"],
        locations: TensorType["locations", 2] = None,
        n_to_find: int = 10, 
        ctx_len: int = 20
    ):
        assert locations is not None

        unique_batch_pos = torch.unique(locations[:,0])
        taken = set(unique_batch_pos.tolist())
        free = []
        value = 0
        while value < n_to_find:
            if value not in taken:
                free.append(value)
                value += 1
        return tokens[free, 10:10+ctx_len]