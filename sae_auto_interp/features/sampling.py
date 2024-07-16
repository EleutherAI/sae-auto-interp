import random
import torch
from ..logger import logger

def split_activation_quantiles(examples, n_quantiles):
    max_activation = examples[0].max_activation
    
    thresholds = [max_activation * i / n_quantiles for i in range(1, n_quantiles)]
    quantiles = [[] for _ in range(n_quantiles)]

    for example in examples:
        for i, threshold in enumerate(thresholds):
            if example.max_activation <= threshold:
                quantiles[i].append(example)
                break
        else:
            quantiles[-1].append(example)
    
    return quantiles

def split_quantiles(examples, n_quantiles):
    n = len(examples)
    quantile_size = n // n_quantiles
    
    return [
        examples[i:i + quantile_size] if i < (n_quantiles - 1) * quantile_size
        else examples[i:]
        for i in range(0, n, quantile_size)
    ]

def check_quantile(quantile, n_test):
    if len(quantile) < n_test:
        logger.error(f"Quantile has too few examples")
        raise ValueError(f"Quantile has too few examples")

def default_sampler(
        record, 
        n_train = 10, 
        n_test = 10
    ):  
    n_samples = n_train + n_test
    samples = random.sample(record.examples, n_samples)
    record.train = samples[:n_train]
    record.test = samples[n_test:]

def sample_activation_quantiles(
    record,
    n_train=10,
    n_test=10,
    n_quantiles=3,
    seed=22,
):
    """
    Split examples into quantiles based on fractions of the max activation across all examples.
    Sample train examples from the first example and test examples from the rest.
    Will return n_quantiles - 1 test sets.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    activation_quantiles = split_activation_quantiles(record.examples, n_quantiles)
    train_examples = random.sample(activation_quantiles[0], n_train)

    test_quantiles = activation_quantiles[1:]
    test_examples = []

    for quantile in test_quantiles:
        check_quantile(quantile, n_test)
        test_examples.append(random.sample(quantile, n_test))

    record.train = train_examples
    record.test = test_examples


def sample_top_and_activation_quantiles(
    record,
    n_train=10,
    n_test=5,
    n_quantiles=4,
    seed=22,
):
    """
    Train examples are the top n_train examples, then split the rest quantiles of the max activation.
    Sample test examples from each quantile.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    # print(record, n_train, n_test, n_quantiles, seed, decode)
    train_examples = record.examples[:n_train]

    activation_quantiles = split_activation_quantiles(
        record.examples[n_train:], n_quantiles
    )

    test_examples = []

    for quantile in activation_quantiles:
        check_quantile(quantile, n_test)
        test_examples.append(random.sample(quantile, n_test))

    record.train = train_examples
    record.test = test_examples

def sample_top_and_quantiles(
    record,
    n_train=10,
    n_test=10,
    n_quantiles=4,
    seed=22,
):
    """
    Train examples are the top n_train examples, then split the rest into
    evenly sized quantiles and sample from each quantile for testing.

    Args:
        record (FeatureRecord): The record to sample from.
        n_train (int): The number of examples to sample for training.
        n_test (int): The number of examples to sample for testing from each quantile.
        n_quantiles (int): The number of quantiles to split the remaining examples into.
        n_extra (int): The number of extra examples to sample from the remaining set.
        seed (int): The random seed to use for reproducibility.

    Returns:
        Tuple[List[Example], List[List[Example]], List[Example]]: A tuple containing the training examples,
        a list of test examples for each quantile, and a list of extra examples (if n_extra > 0).
    """
    random.seed(seed)
    torch.manual_seed(seed)

    examples = record.examples

    # Sample n_train examples for training
    train_examples = examples[:n_train]
    remaining_examples = examples[n_train:]

    quantiles = split_quantiles(remaining_examples, n_quantiles)

    test_examples = []

    for quantile in quantiles:
        check_quantile(quantile, n_test)
        test_examples.append(random.sample(quantile, n_test))

    record.train = train_examples
    record.test = test_examples