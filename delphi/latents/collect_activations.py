from typing import Any
from contextlib import contextmanager
from torch import nn, Tensor
from collections import defaultdict

import torch



@contextmanager
def collect_activations(model: nn.Module, hookpoints: list[str]):
    """
    Context manager that temporarily hooks models and collects their activations.
    An activation tensor is produced for each batch processed and stored in a list for that hookpoint
    in the activations dictionary.
    
    Args:
        model: The transformer model to hook
        hookpoints: List of hookpoints to collect activations from
        
    Yields:
        Dictionary mapping hookpoints to their collected activations
    """
    activations = defaultdict(list)
    handles = []
    
    def create_hook(hookpoint: str):
        def hook_fn(module: nn.Module, input: Any, output: Tensor) -> Tensor:
            # If output is a tuple (like in some transformer layers), take first element
            if isinstance(output, tuple):
                activations[hookpoint].append(output[0])
            else:
                activations[hookpoint].append(output)
        return hook_fn

    for name, module in model.named_modules():
        if name in hookpoints:
            handle = module.register_forward_hook(create_hook(name))
            handles.append(handle)
    
    try:
        yield activations
    except Exception as e:
        for handle in handles:
            handle.remove()
        raise e
    finally:
        for handle in handles:
            handle.remove()
        
        for hookpoint in activations:
            if activations[hookpoint]:
                activations[hookpoint] = torch.stack(activations[hookpoint])
    
