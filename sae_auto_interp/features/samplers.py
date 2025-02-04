import random
from collections import deque
from typing import Literal, cast
from torchtyping import TensorType

from ..config import ExperimentConfig
from .features import Example, FeatureRecord
from ..logger import logger

            
def split_activation_quantiles(examples: list[Example], n_quantiles: int, n_samples: int, seed: int = 22):
    """
    TODO review this, there is a possible bug here: `examples[0].max_activation < threshold`

    Split the examples into n_quantiles and sample n_samples from each quantile.

    Args:
        examples: list of Examples, assumed to be in descending sorted order by max_activation
        n_quantiles: number of quantiles to split the examples into
        n_samples: number of samples to sample from each quantile
        seed: seed for the random number generator

    Returns:
        list of lists of Examples, each inner list contains n_samples from a unique quantile
    """
    random.seed(seed)

    examples = deque(examples)
    max_activation = examples[0].max_activation

    # For 4 quantiles, thresholds are 0.25, 0.5, 0.75
    thresholds = [max_activation * i / n_quantiles for i in range(1, n_quantiles)]

    samples: list[list[Example]] = []
    for threshold in thresholds:
        # Get all examples in quantile
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
    """
    Randomly select (n_samples // n_quantiles) samples from each quantile.
    """
    random.seed(seed)

    quantile_size = len(examples) // n_quantiles
    samples_per_quantile = n_samples // n_quantiles
    samples: list[list[Example]] = []
    for i in range(n_quantiles):
        # Take an evenly spaced slice of the examples for the quantile
        quantile = examples[i * quantile_size : (i + 1) * quantile_size]

        # Take a random sample of the examples. If there are less than samples_per_quantile, use all samples
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
                logger.warning(f"n_train is greater than the number of examples, using all examples")
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
                    example.normalized_activations = cast(TensorType["seq"], (example.activations * 10 / max_activation).floor())
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

    
    