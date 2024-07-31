import random
from typing import List

from ..logger import logger
from .features import Example, FeatureRecord


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
        examples[i * quantile_size : (i + 1) * quantile_size]
        for i in range(n_quantiles)
    ]


def check_quantile(quantile, n_test):
    if len(quantile) < n_test:
        logger.error("Quantile has too few examples")
        raise ValueError("Quantile has too few examples")


def random_and_activation_quantiles(
    record: FeatureRecord,
    n_train=10,
    n_test=10,
    n_quantiles=3,
    seed=22,
):
    random.seed(seed)

    activation_quantiles = split_activation_quantiles(record.examples, n_quantiles)
    train_examples = random.sample(activation_quantiles[0], n_train)

    test_quantiles = activation_quantiles[1:]
    test_examples = []

    for quantile in test_quantiles:
        check_quantile(quantile, n_test)
        test_examples.append(random.sample(quantile, n_test))

    record.train = train_examples
    record.test = test_examples


def top_and_activation_quantiles(
    record: FeatureRecord,
    n_train=10,
    n_test=5,
    n_quantiles=4,
    seed=22,
):
    random.seed(seed)

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


def top_sample(
    record: FeatureRecord,
    n_examples=10,
    type="train",
    seed=22,
):
    random.seed(seed)

    examples = record.examples[:n_examples]

    if type == "train":
        record.train = examples
    else:
        record.test = examples


def random_sample(
    record: FeatureRecord,
    n_examples=10,
    type="train",
    seed=22,
):
    random.seed(seed)

    examples = random.sample(record.examples, n_examples)

    if type == "train":
        record.train = examples
    else:
        record.test = examples


def quantiles_sample(
    record: FeatureRecord,
    n_quantiles=4,
    n_examples=10,
    type="train",
    seed=22,
):
    random.seed(seed)

    quantiles = split_quantiles(record.examples, n_quantiles)
    examples = []
    for quantile in quantiles:
        check_quantile(quantile, n_examples)
        examples.append(random.sample(quantile, n_examples))

    if type == "train":
        record.train = examples
    else:
        record.test = examples


def top_and_quantiles(
    record: FeatureRecord,
    n_train=10,
    n_test=10,
    n_quantiles=4,
    seed=22,
):
    random.seed(seed)

    examples = record.examples

    train_examples = examples[:n_train]

    quantiles = split_quantiles(examples[n_train:], n_quantiles)

    test_examples = []

    for quantile in quantiles:
        check_quantile(quantile, n_test)
        test_examples.append(random.sample(quantile, n_test))

    record.train = train_examples
    record.test = test_examples


def random_and_quantiles(
    record: FeatureRecord,
    n_train=10,
    n_test=10,
    n_quantiles=4,
    seed=22,
):
    random.seed(seed)

    examples = record.examples

    train_examples = random.sample(examples, n_train)

    remaining_examples = [
        example for example in examples if example not in train_examples
    ]

    quantiles = split_quantiles(remaining_examples, n_quantiles)

    test_examples = []

    for quantile in quantiles:
        check_quantile(quantile, n_test)
        test_examples.append(random.sample(quantile, n_test))

    record.train = train_examples
    record.test = test_examples
