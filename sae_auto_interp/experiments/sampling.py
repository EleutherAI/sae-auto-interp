from typing import List, Tuple
import random
from ..features.features import Example, FeatureRecord
from ..logger import logger

def sample_top_and_quantiles(
    record: FeatureRecord, 
    n_train: int = 10,
    train_population: int = 50, 
    n_test_per_quantile: int = 10, 
    n_quantiles: int = 5, 
    seed=22
) -> Tuple[List[Example], List[List[Example]]]:
    """
    Return one train and n_quantiles test sets for a given record.
    Samples randomly from the train_population to build the train set.
    Samples randomly per quantile to build the test sets.

    Args:
        record (FeatureRecord): The record to sample from.
        n_train (int, optional): Number of examples in the train set. Defaults to 10.
        train_population (int, optional): Number of examples to sample from to build the train set. Defaults to 50.
        n_test_per_quantile (int, optional): Number of examples in each quantile. Defaults to 10.
        n_quantiles (int, optional): Number of quantiles. Defaults to 5.
        seed (int, optional): Random seed. Defaults to 22.

    Returns:
        Tuple[List[Example], List[List[Example]]]: Train set and test sets.
    """
    random.seed(seed)
    if n_train + (n_test_per_quantile * n_quantiles) > len(record.examples):
        logger.info(f"Not enough examples for {record.feature}")
        raise ValueError
    
    train_set = random.sample(
        record.examples[:train_population], 
        n_train
    )

    remaining_examples = record.examples[train_population:]

    # Divide remaining examples into quantiles
    quantile_size = len(remaining_examples) // n_quantiles

    if quantile_size < n_test_per_quantile:
        logger.info(f"Not enough examples per quantile for {record.feature}")
        raise ValueError

    quantiles = [
        remaining_examples[i * quantile_size:(i + 1) * quantile_size] 
        for i in range(n_quantiles)
    ]

    test_sets = []
    for quantile in quantiles:
        test_set = random.sample(quantile, n_test_per_quantile)
        test_sets.append(test_set)

    return train_set, test_sets
