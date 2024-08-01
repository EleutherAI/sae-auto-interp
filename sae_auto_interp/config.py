from dataclasses import dataclass

from simple_parsing import Serializable


@dataclass
class ExperimentConfig(Serializable):
    train_type: str = "top"
    """Type of sampler to use"""

    n_examples_train: int = 10
    """Number of examples to sample for training"""

    n_examples_test: int = 5
    """Number of examples to sample for testing"""

    n_quantiles: int = 10
    """Number of quantiles to sample"""

    n_random: int = 5
    """Number of random examples to sample"""


@dataclass
class FeatureConfig(Serializable):
    width: int = 131072
    """Number of features in the autoencoder"""

    min_examples: int = 200
    """Minimum number of examples for a feature to be included"""

    max_examples: int = 10000
    """Maximum number of examples for a feature to included"""

    example_ctx_len: int = 64
    """Length of each example"""

    n_splits: int = 5
    """Number of splits that features were devided into"""


@dataclass
class CacheConfig(Serializable):
    batch_size: int = 32
    """Number of sequences to process in a batch"""

    ctx_len: int = 256
    """Context length of the autoencoder. Each batch is shape (batch_size, ctx_len)"""

    n_tokens: int = 10_000_000
    """Number of tokens to cache"""

    n_splits: int = 5
    """Number of splits to divide .safetensors into"""
