from typing import List, Tuple
import random
from ..features.features import Example, FeatureRecord
from ..logger import logger
import torch

def sample_quantiles(
    record,
    n_train=10,
    n_test=5,
    n_quantiles=2,
    seed=22,
):
    """
    Split a record's examples into quantiles based on fractions 
    of the max activation and sample from each quantile.

    Args:
        record (FeatureRecord): The record to sample from.
        n_train (int): The number of examples to sample for training.
        n_test (int): The number of examples to sample for testing.
        n_quantiles (int): The number of quantiles to split the examples into.
        seed (int): The random seed to use for reproducibility.

    Returns:
        Tuple[List[Example], List[List[Example]]]: A tuple containing the training examples
        and a list of test examples for each quantile.
    """
    random.seed(seed)
    torch.manual_seed(seed)  # Also set torch seed for reproducibility

    # Extract max activations from examples
    max_activations = torch.tensor([example.max_activation for example in record.examples])
    

    # Calculate thresholds based on fractions of the overall maximum activation
    overall_max = max_activations.max().item()
    thresholds = [overall_max * (i + 1) / n_quantiles for i in range(n_quantiles - 1)]
    
    quantiles = []
    start = 0
    for i, end in enumerate(thresholds):
        # Filter examples in this quantile
        quantile_examples = [
            example for example in record.examples
            if start <= example.max_activation < end
        ]
        
        n_samples = n_train if i == 0 else n_test

        if len(quantile_examples) < n_samples:
            raise ValueError(f"Quantile {i} has too few examples in record {record}")
        
        quantile_sample = random.sample(quantile_examples, n_samples)
        quantiles.append(quantile_sample)
        
        start = end
    
    # Add the last quantile
    last_quantile = [
        example for example in record.examples
        if example.max_activation >= start
    ]
    n_samples = min(n_test, len(last_quantile))
    quantiles.append(random.sample(last_quantile, n_samples))
    
    return quantiles[0], quantiles[1:]