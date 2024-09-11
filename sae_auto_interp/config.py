from dataclasses import dataclass
from typing import Literal

from simple_parsing import Serializable


@dataclass
class ExperimentConfig(Serializable):
    
    n_examples_train: int = 40
    """Number of examples to sample for training"""

    example_ctx_len: int = 20
    """Length of each example"""

    n_examples_test: int = 5
    """Number of examples to sample for testing"""

    n_quantiles: int = 20
    """Number of quantiles to sample"""

    n_random: int = 50
    """Number of random examples to sample"""

    train_type: Literal["top", "random", "quantiles"] = "random"
    """Type of sampler to use for training"""

    test_type: Literal["even", "activation"] = "even"
    """Type of sampler to use for testing"""




@dataclass
class FeatureConfig(Serializable):
    width: int = 131072
    """Number of features in the autoencoder"""

    min_examples: int = 200
    """Minimum number of examples for a feature to be included"""

    max_examples: int = 10000
    """Maximum number of examples for a feature to included"""

    n_splits: int = 5
    """Number of splits that features were devided into"""


@dataclass
class CacheConfig(Serializable):

    dataset_repo: str = "kh4dien/fineweb-100m-sample"
    """Dataset repository to use"""

    dataset_split: str = "train"
    """Dataset split to use""" 

    dataset_name: str = ""
    """Dataset name to use"""

    batch_size: int = 32
    """Number of sequences to process in a batch"""

    ctx_len: int = 256
    """Context length of the autoencoder. Each batch is shape (batch_size, ctx_len)"""

    n_tokens: int = 10_000_000
    """Number of tokens to cache"""

    n_splits: int = 5
    """Number of splits to divide .safetensors into"""
