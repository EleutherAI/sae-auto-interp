import random
from collections import deque
from typing import Literal

from ..config import ExperimentConfig
from ..logger import logger
from .features import Example, FeatureRecord


def split_activation_quantiles(
    examples: list[Example], 
    n_quantiles: int,
    n_samples: int,
    seed: int = 22
):
    random.seed(seed)

    max_activation = examples[0].max_activation
    thresholds = [max_activation * i / n_quantiles for i in range(1, n_quantiles)]

    samples = []
    examples = deque(examples)

    for threshold in thresholds:
        quantile = []
        while examples and examples[0].max_activation < threshold:
            quantile.append(examples.popleft())

        sample = random.sample(quantile, n_samples)
        samples.append(sample)

    sample = random.sample(examples, n_samples)
    samples.append(sample)
    
    return samples


def split_quantiles(
    examples: list[Example], 
    n_quantiles: int, 
    n_samples: int,
    seed: int = 22
):
    random.seed(seed)

    quantile_size = len(examples) // n_quantiles
    samples_per_quantile = max(1, n_samples // n_quantiles)
    samples = []
    for i in range(n_quantiles):
        quantile = examples[i * quantile_size : (i + 1) * quantile_size]
        if len(quantile) < samples_per_quantile:
            sample = quantile
            logger.info(f"Quantile {i} has less than {samples_per_quantile} samples, using all samples")
        else:
            sample = random.sample(quantile, samples_per_quantile)
        samples.append(sample)

    return samples


def train(
    examples: list[Example],
    max_activation: float,
    n_train: int,
    train_type: Literal["top", "random","quantiles"],
    n_quantiles: int = 10,
    seed: int = 22,
):
    match train_type:
        case "top":
            selected_examples = examples[:n_train]
            for example in selected_examples:
                example.normalized_activations = (example.activations * 10 / max_activation).floor()
            return selected_examples
        case "random":
            random.seed(seed)
            if n_train > len(examples):
                logger.warning("n_train is greater than the number of examples, using all examples")
                for example in examples:
                    example.normalized_activations = (example.activations * 10 / max_activation).floor()
                return examples
            selected_examples = random.sample(examples, n_train)
            for example in selected_examples:
                example.normalized_activations = (example.activations * 10 / max_activation).floor()
            return selected_examples
        case "quantiles":
            
            selected_examples_quantiles = split_quantiles(examples, n_quantiles, n_train)
            selected_examples = []
            for quantile in selected_examples_quantiles:
                for example in quantile:
                    example.normalized_activations = (example.activations * 10 / max_activation).floor()
                selected_examples.extend(quantile)
            return selected_examples


def test(
    examples: list[Example],
    max_activation: float,
    n_test: int,
    n_quantiles: int,
    test_type: Literal["quantiles", "activation"],
    
):
    match test_type:
        case "quantiles":
            selected_examples = split_quantiles(examples, n_quantiles, n_test)
            for quantile in selected_examples:
                for example in quantile:
                    example.normalized_activations = (example.activations * 10 / max_activation).floor()
            return selected_examples
        case "activation":
            selected_examples = split_activation_quantiles(examples, n_quantiles, n_test)
            for quantile in selected_examples:
                for example in quantile:
                    example.normalized_activations = (example.activations * 10 / max_activation).floor()
            return selected_examples


def sample(
    record: FeatureRecord,
    cfg: ExperimentConfig,
):
    examples = record.examples
    max_activation = record.max_activation
    _train = train(
        examples,
        max_activation,
        cfg.n_examples_train,
        cfg.train_type,
        n_quantiles=cfg.n_quantiles,
    )
    record.train = _train
    if cfg.n_examples_test > 0: 
        _test = test(
            examples,
            max_activation,
            cfg.n_examples_test,
            cfg.n_quantiles,
            cfg.test_type,   
        )
        record.test = _test
