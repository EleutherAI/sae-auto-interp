import torch




def unravel_index(flat_index, shape):

    indices = []
    for dim_size in reversed(shape):
        indices.append(flat_index % dim_size)
        flat_index = flat_index // dim_size
    return tuple(reversed(indices))


def topk(tensor, k):

    flat_tensor = tensor.flatten()

    top_values, flat_indices = torch.topk(flat_tensor, k)

    original_indices = [unravel_index(idx.item(), tensor.size()) for idx in flat_indices]

    return top_values.tolist(), original_indices

def find_smallest_index_above_zero(arr):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] > 0:
            left = mid + 1
        else:
            right = mid - 1
    if right >= 0 and arr[right] > 0:
        return right
    return -1

