from typing import List, Tuple
import random
from ..features.features import Example, FeatureRecord

def sample_top_and_quantiles(
    record: FeatureRecord, 
    n_train: int = 10, 
    n_test: int = 10, 
    n_quantiles: int = 10, 
    seed=22
) -> List[Tuple[List[Example], List[Example]]]:
    """
    Return multiple (train, test) sets. Train sets are always the top n_train examples,
    and each test set is sampled from a different quantile.

    Args:
    n_train (int): Number of top examples to use as train set in each pair.
    n_test (int): Number of examples to sample for test set in each pair.
    n_quantiles (int, optional): Number of quantiles to divide the remaining examples into. Defaults to 4.

    Returns:
    List[Tuple[List[Example], List[Example]]]: List of (train_set, test_set) pairs.
    """
    random.seed(seed)
    if n_train + (n_test * n_quantiles) > len(record.examples):
        raise ValueError(f"Not enough examples to create {n_quantiles} test sets of size {n_test} after using {n_train} for training")

    train_set = record.examples[:n_train]
    remaining_examples = record.examples[n_train:]

    # Divide remaining examples into quantiles
    quantile_size = len(remaining_examples) // n_quantiles
    quantiles = [remaining_examples[i * quantile_size:(i + 1) * quantile_size] for i in range(n_quantiles)]

    results = []
    for quantile in quantiles:
        if len(quantile) < n_test:
            raise ValueError(f"Quantile size ({len(quantile)}) is smaller than n_test ({n_test})")
        
        test_set = random.sample(quantile, n_test)
        results.append((train_set, test_set))

    return results


def sample_top_and_quantiles_single(
    record: FeatureRecord, 
    n_train: int = 10, 
    n_test: int = 10, 
    n_quantiles: int = 10, 
    seed=22
) -> Tuple[List[Example], List[List[Example]]]:
    """
    Return one train set and multiple test sets. The train set is always the top n_train examples,
    and each test set is sampled from a different quantile.

    Args:
    n_train (int): Number of top examples to use as train set.
    n_test (int): Number of examples to sample for each test set.
    n_quantiles (int, optional): Number of quantiles to divide the remaining examples into. Defaults to 10.

    Returns:
    Tuple[List[Example], List[List[Example]]]: A tuple containing the train set and a list of test sets.
    """
    random.seed(seed)
    if n_train + (n_test * n_quantiles) > len(record.examples):
        raise ValueError(f"Not enough examples to create {n_quantiles} test sets of size {n_test} after using {n_train} for training")

    train_set = record.examples[:n_train]
    remaining_examples = record.examples[n_train:]

    # Divide remaining examples into quantiles
    quantile_size = len(remaining_examples) // n_quantiles
    quantiles = [remaining_examples[i * quantile_size:(i + 1) * quantile_size] for i in range(n_quantiles)]

    test_sets = []
    for quantile in quantiles:
        if len(quantile) < n_test:
            raise ValueError(f"Quantile size ({len(quantile)}) is smaller than n_test ({n_test})")
        
        test_set = random.sample(quantile, n_test)
        test_sets.append(test_set)

    return [(train_set, test_sets)]


def sample_top_and_random(
    record: FeatureRecord, 
    n_train: int = 10, 
    n_test: int = 10, 
    seed=22
) -> List[Tuple[List[Example], List[Example]]]:
    """
    Sample top examples as train set and random samples from the rest as test set.

    Args:
    n_train (int): Number of top examples to use as train set.
    n_test (int): Number of random examples to sample for test set.

    Returns:
    Tuple[List[Example], List[Example]]: (train_set, test_set)
    """
    random.seed(seed)
    if n_train + n_test > len(record.examples):
        raise ValueError("n_train + n_test cannot exceed the total number of examples")

    train_set = record.examples[:n_train]
    remaining_examples = record.examples[n_train:]
    test_set = random.sample(remaining_examples, n_test)

    return [(train_set, test_set)]
