import random
import torch

from typing import List
from features import Example
from ..logger import logger

def split_activation_quantiles(examples: List[Example], n_quantiles: int):
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


def split_quantiles(examples: List[Example], n_quantiles: int):
    n = len(examples)
    quantile_size = n // n_quantiles
    
    return [
        examples[i * quantile_size:(i + 1) * quantile_size] 
        for i in range(n_quantiles)
    ]



def get_extra_examples(record, n_extra, train_examples, test_examples):
    used_examples = set(
        example 
        for example in train_examples + sum(test_examples, [])
    )

    remaining_examples = [
        example 
        for example in record.examples 
        if example not in used_examples
    ]

    if len(remaining_examples) < n_extra:
        logger.error(f"Not enough extra examples in {record.feature}.")
        raise ValueError(f"Not enough extra examples in {record.feature}.")
    
    extra_examples = random.sample(
        remaining_examples, 
        n_extra
    )

    return extra_examples


def sample_activation_quantiles(
    record,
    n_train=10,
    n_test=10,
    n_quantiles=3,
    n_extra=0,
    seed=22,
):
    """
    Split examples into quantiles based on fractions of the max activation across all examples.
    Sample train examples from the first example and test examples from the rest.
    Will return n_quantiles - 1 test sets.
    """
    random.seed(seed)
    torch.manual_seed(seed)  # Also set torch seed for reproducibility

    activation_quantiles = split_activation_quantiles(record.examples, n_quantiles)
    train_examples = random.sample(activation_quantiles[0], n_train)

    test_quantiles = activation_quantiles[1:]
    test_examples = []

    for quantile in test_quantiles:
        if len(quantile) < n_test:
            logger.error(f"Quantile has too few examples in {record.feature}")
            raise ValueError(f"Quantile has too few examples in {record.feature}")
        
        test_examples.append(random.sample(quantile, n_test))

    if n_extra > 0:
        extra_examples = get_extra_examples(
            record, 
            n_extra, 
            train_examples, 
            test_examples
        )
        return train_examples, test_examples, extra_examples
    else:
        return train_examples, test_examples


def sample_top_and_activation_quantiles(
    record,
    n_train=10,
    n_test=5,
    n_quantiles=4,
    n_extra=0,
    seed=22,
):
    """
    Train examples are the top n_train examples, then split the rest quantiles of the max activation.
    Sample test examples from each quantile.
    """
    random.seed(seed)
    torch.manual_seed(seed)  # Also set torch seed for reproducibility

    train_examples = record.examples[:n_train]

    activation_quantiles = split_activation_quantiles(record.examples[n_train:], n_quantiles)

    test_examples = []

    for quantile in activation_quantiles:
        if len(quantile) < n_test:
            logger.error(f"Quantile has too few examples in {record.feature}")
            raise ValueError(f"Quantile has too few examples in {record.feature}")
        
        test_examples.append(random.sample(quantile, n_test))

    if n_extra > 0:
        extra_examples = get_extra_examples(
            record, 
            n_extra, 
            train_examples, 
            test_examples
        )
        return train_examples, test_examples, extra_examples
    else:
        return train_examples, test_examples


def sample_top_and_quantiles(
    record,
    n_train=10,
    n_test=5,
    n_quantiles=2,
    n_extra=0,
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
    torch.manual_seed(seed)  # Also set torch seed for reproducibility

    if len(record.examples) < n_train + (n_test * n_quantiles):
        logger.error(f"Not enough examples in {record.feature} for the requested sampling")
        raise ValueError(f"Not enough examples in {record.feature} for the requested sampling")

    examples = record.examples

    # Sample n_train examples for training
    train_examples = examples[:n_train]
    remaining_examples = examples[n_train:]

    quantiles = split_quantiles(remaining_examples, n_quantiles)
    # print(len(quantiles))
    # print(len(quantiles[0]))
    # print(len(quantiles[-1]))
    test_examples = []

    for quantile in quantiles:
        if len(quantile) < n_test:
            logger.error(f"Quantile has too few examples in {record.feature}")
            raise ValueError(f"Quantile has too few examples in {record.feature}")
        examples = random.sample(quantile, n_test)
        for example in examples:
            example.decode(record.tokenizer)
        test_examples.append(examples)
    
    extra_examples = []
    if n_extra > 0:
        extra_examples = get_extra_examples(
            record, 
            n_extra, 
            train_examples, 
            test_examples
        )
        return train_examples, test_examples, extra_examples
    else:
        return train_examples, test_examples