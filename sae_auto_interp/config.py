from dataclasses import dataclass
from typing import Literal

from simple_parsing import Serializable


@dataclass
class ExperimentConfig(Serializable):
    
    n_examples_train: int = 25
    """Number of examples to sample for training"""

    n_examples_test: int = 25
    """Number of examples to sample for testing"""

    n_quantiles: int = 5
    """Number of quantiles to sample"""

    example_ctx_len: int = 32
    """Length of each example"""

    n_random: int = 50
    """Number of random examples to sample"""

    train_type: Literal["top", "random", "quantiles"] = "quantiles"
    """Type of sampler to use for training"""

    test_type: Literal["quantiles", "activation"] = "quantiles"
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

    dataset_column_name: str = "text"
    """Dataset column name to use"""

    batch_size: int = 32
    """Number of sequences to process in a batch"""

    ctx_len: int = 256
    """Context length of the autoencoder. Each batch is shape (batch_size, ctx_len)"""

    n_tokens: int = 10_000_000
    """Number of tokens to cache"""

    n_splits: int = 5
    """Number of splits to divide .safetensors into"""
