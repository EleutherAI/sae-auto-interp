import torch
from torchtyping import TensorType

from .features import FeatureRecord, prepare_examples
from .loader import BufferOutput
from ..config import FeatureConfig

from time import time


def _to_dense(tokens, activations, locations):
    # Reconstruct dense tensor
    num_sequences, seq_len = tokens.shape
    start_time = time()
    
    sparse_activations = torch.sparse_coo_tensor(
        locations.t(), activations, (num_sequences, seq_len)
    )
    #print(f"Coo tensor: {time() - start_time}")
    dense_activations = sparse_activations.to_dense()
    #print(f"To dense: {time() - start_time}")

    # Get unique location rows along the tokens tensor
    
    start_time = time()
    unique_sequence_pos = torch.unique_consecutive(locations[:, 0])
    #print(f"Unique: {time() - start_time}")
    start_time = time()
    token_batches = tokens[unique_sequence_pos]
    #max_buffer = torch.zeros(num_contexts,dtype=buffer_output.activations.dtype).scatter_reduce_(0, ctx_indices, buffer_output.activations, reduce='amax')
    
    dense_activations = dense_activations[unique_sequence_pos]
    #print(f"Indexing: {time() - start_time}")
    #token_batches = tokens
    
    return token_batches, dense_activations


# TODO: We should add an option to change stride size
def _reconstruct_examples(dense_activations, token_batches, ctx_len):
    # Max pool activations
    avg_pools = torch.nn.functional.max_pool1d(
        dense_activations, kernel_size=ctx_len, stride=ctx_len
    )

    # Unfold tokens and activations to match
    activation_windows = dense_activations.unfold(1, ctx_len, ctx_len).reshape(
        -1, ctx_len
    )
    token_windows = token_batches.unfold(1, ctx_len, ctx_len).reshape(-1, ctx_len)

    return token_windows, activation_windows, avg_pools


def _top_k_pools_old(dense_activations, token_batches, ctx_len, max_examples):
    start_time = time()
    token_windows, activation_windows, avg_pools = \
        _reconstruct_examples(dense_activations, token_batches, ctx_len)
    #print(f"Reconstruct: {time() - start_time}")
    # # Filter out zero pools
    non_zero_mask = avg_pools != 0
    non_zero_pools = avg_pools[non_zero_mask]

    # Get top k activation pools
    start_time = time()
    k = min(max_examples,len(non_zero_pools))
    top_values,top_indices = torch.topk(avg_pools.flatten(), k,sorted=True)
    #print(f"Top k: {time() - start_time}")
    #print(top_values[:10])
    
    # Get the top indices
    activation_windows = activation_windows[top_indices]
    token_windows = token_windows[top_indices]
    return token_windows, activation_windows


def _top_k_pools(max_buffer,split_activations, buffer_tokens, ctx_len, max_examples):

    
    # Get top k activation pools
    k = min(max_examples, len(max_buffer))
    #start_time = time()
    top_values,top_indices = torch.topk(max_buffer, k,sorted=True)
    #print(f"Top k: {time() - start_time}")
    # Get the top indices
    activation_windows = torch.stack([split_activations[i] for i in top_indices])
    token_windows = buffer_tokens[top_indices]

    return token_windows, activation_windows

def pool_max_activation_windows_old(
    record,
    buffer_output: BufferOutput,
    tokens: TensorType["batch", "seq"],
    ctx_len: int,
    max_examples: int,
):
    start_time = time()
    start_start = time()
    token_batches, dense_activations = _to_dense(
        tokens, buffer_output.activations, buffer_output.locations
    )
    #print(f"To dense: {time() - start_time}")
    start_time = time()
    token_windows, activation_windows = _top_k_pools_old(
        dense_activations, token_batches, ctx_len, max_examples
    )
    #print(f"Top k: {time() - start_time}")
    #print(f"Total: {time() - start_start}")
    # Set as examples
    start_time = time()

    record.examples = prepare_examples(token_windows, activation_windows)
    #print(f"Prepare examples: {time() - start_time}")
    


def pool_max_activation_windows(
    record,
    buffer_output: BufferOutput,
    tokens: TensorType["batch", "seq"],
    cfg: FeatureConfig,
):
    #start_time = time()
    #start_start = time()
    #buffer_output.activations [occurences]
    #buffer_output.locations [occurences, 2] -> batch,seq -> new_batch
    buffer_output.activations.to(torch.float32)
    flat_indices = buffer_output.locations[:, 0] * tokens.shape[1] + buffer_output.locations[:, 1]
    #print(flat_indices[:10])
    num_contexts = tokens.numel()//ctx_len
    ctx_indices = flat_indices//ctx_len
    index_within_ctx = flat_indices%ctx_len
    unique_ctx_indices,inverses,lengths = torch.unique_consecutive(ctx_indices,return_counts=True,return_inverse=True)
    max_buffer = torch.segment_reduce(buffer_output.activations, 'max', lengths=lengths)
    #split_activations = torch.split(buffer_output.activations, lengths.tolist())
    #split_index_within_ctx = torch.split(index_within_ctx, lengths.tolist())
    
    new_tensor=torch.zeros(len(unique_ctx_indices),ctx_len,dtype=buffer_output.activations.dtype)
    new_tensor[inverses,index_within_ctx]=buffer_output.activations
    # new_tensor=[]
    # for non_zero_acts,non_zero_indices in zip(split_activations,split_index_within_ctx):
    #     zero_tensor = torch.zeros(ctx_len,dtype=non_zero_acts.dtype)
    #     zero_tensor[non_zero_indices] = non_zero_acts
    #     new_tensor.append(zero_tensor)
    # new_tensor = torch.stack(new_tensor)
    #print(new_tensor.shape)
    #max_buffer = torch.zeros(num_contexts,dtype=buffer_output.activations.dtype).scatter_reduce_(0, ctx_indices, buffer_output.activations, reduce='amax')
    #print(max_buffer.shape)
    #print(unique_ctx_indices.shape)
    buffer_tokens = tokens.reshape(-1,ctx_len)
    buffer_tokens = buffer_tokens[unique_ctx_indices]
    #print(max_buffer[:10])
    #print(max_buffer.shape)


    # token_batches, dense_activations = _to_dense(
    #     tokens, buffer_output.activations, buffer_output.locations
    # )
    #print(f"To dense: {time() - start_time}")
    #start_time = time()
    token_windows, activation_windows = _top_k_pools(
        max_buffer,new_tensor, buffer_tokens, ctx_len, max_examples
    )
    #print(f"Top k: {time() - start_time}")
    #total= time() - start_start
    #print(token_windows.shape)
    #print(activation_windows.shape)
    #print(token_windows[:10])
    #print(activation_windows[:10])
    # Set as examples
    #start_time = time()
    # batch,seq,2
    record.examples = prepare_examples(token_windows, activation_windows)
    #print(record.examples[:5])
    #print(record.examples[-1])
    #print(f"Prepare examples: {time() - start_time}")
    
    

def random_activation_windows(
    record,
    tokens: TensorType["batch", "seq"],
    buffer_output: BufferOutput,
    ctx_len: int,
    n_random: int,
):
    torch.manual_seed(22)
    batch_size = tokens.shape[0]
    start_time = time()
    unique_batch_pos = buffer_output.locations[:, 0].unique()
    #print(f"Unique batch pos: {time() - start_time} seconds")

    #start_time = time()
    mask = torch.ones(batch_size, dtype=torch.bool)
    mask[unique_batch_pos] = False
    #print(f"Mask: {time() - start_time} seconds")

    available_indices = mask.nonzero().squeeze()

    #start_time = time()
    #print(available_indices.shape)
    # selected_indices = available_indices[
    #     torch.randperm(len(available_indices))[:n_random]
    # ]
    selected_indices = available_indices[torch.randint(0,len(available_indices),size=(n_random,))]
    #print(f"Perm: {time() - start_time} seconds")
    #start_time = time()
    toks = tokens[selected_indices, 10 : 10 + ctx_len]
    #print(f"Toks: {time() - start_time} seconds")
    
    #print(f"Prepare examples: {time() - start_time} seconds")
    #start_time = time()
    record.random_examples = prepare_examples(
        toks,
        torch.zeros_like(toks),
    )
    #print(f"Prepare examples: {time() - start_time} seconds")

def default_constructor(
    record: FeatureRecord,
    tokens: TensorType["batch", "seq"],
    buffer_output: BufferOutput,
    n_random: int,
    ctx_len: int,
    max_examples: int,
):
    
    
    pool_max_activation_windows(
        record,
        tokens=tokens,
        buffer_output=buffer_output,
        ctx_len=ctx_len,
        max_examples=max_examples,
    )
    # print(f"Pool max activation windows: {time() - start_time} seconds")
    # start_time = time()
    # pool_max_activation_windows_old(
    #     record,
    #     tokens=tokens,
    #     buffer_output=buffer_output,
    #     ctx_len=ctx_len,
    #     max_examples=max_examples,
    # )
    #print(f"Pool max activation windows old: {time() - start_time} seconds")
    #start_time = time()
    random_activation_windows(
        record,
        tokens=tokens,
        buffer_output=buffer_output,
        n_random=n_random,
        ctx_len=ctx_len,
    )
    #print(f"Random activation windows: {time() - start_time} seconds")
