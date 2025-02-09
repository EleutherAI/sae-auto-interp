from dataclasses import dataclass
from typing import Literal

from simple_parsing import Serializable


@dataclass
class ExperimentConfig(Serializable):
    n_examples_train: int = 40
    """Number of examples to sample for feature explanation generation."""

    n_examples_test: int = 50
    """Number of examples to sample for feature explanation testing."""

    n_quantiles: int = 10
    """Number of feature activation quantiles to sample."""

    example_ctx_len: int = 32
    """Length of each sampled example sequence. Longer sequences 
    reduce detection scoring performance in weak models."""

    n_random: int = 50
    """Number of random examples to sample."""

    train_type: Literal["top", "random", "quantiles"] = "quantiles"
    """Type of sampler to use for feature explanation generation."""

    test_type: Literal["quantiles", "activation"] = "quantiles"
    """Type of sampler to use for feature explanation testing."""


@dataclass
class FeatureConfig(Serializable):
    width: int = 131_072
    """Number of features in each autoencoder"""

    min_examples: int = 200
    """Minimum number of examples for a feature to be explained and scored."""

    max_examples: int = 10_000
    """Maximum number of examples to generate for a single feature."""

    n_splits: int = 5
    """Number of splits that features will be divided into."""


@dataclass
class CacheConfig(Serializable):
    dataset_repo: str = "EleutherAI/rpj-v2-sample"
    """Dataset repository to use for generating feature activations."""

    dataset_split: str = "train[:1%]"
    """Dataset split to use for generating feature activations.""" 

    dataset_name: str = ""
    """Dataset name to use."""

    dataset_row: str = "raw_content"
    """Dataset row to use."""

    batch_size: int = 32
    """Number of sequences to process in a batch."""

    ctx_len: int = 256
    """Context length of the autoencoder. Each batch is shape (batch_size, ctx_len)."""

    n_tokens: int = 10_000_000
    """Number of tokens to cache."""

    n_splits: int = 5
    """Number of splits to divide .safetensors into."""
