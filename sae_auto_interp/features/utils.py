import torch




def prepare_example(example, max_activation=0.0):
    delimited_string = ""
    activation_threshold = max_activation

    pos = 0
    while pos < len(example.tokens):
        if (
            pos + 1 < len(example.tokens)
            and example.activations[pos + 1] > activation_threshold
        ):
            delimited_string += example.str_toks[pos]
            pos += 1
        elif example.activations[pos] > activation_threshold:
            delimited_string += "<<"

            seq = ""
            while (
                pos < len(example.tokens)
                and example.activations[pos] > activation_threshold
            ):

                delimited_string += example.str_toks[pos]
                seq += example.str_toks[pos]
                pos += 1

            delimited_string += ">>"

        else:
            delimited_string += example.str_toks[pos]
            pos += 1

    return delimited_string

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

