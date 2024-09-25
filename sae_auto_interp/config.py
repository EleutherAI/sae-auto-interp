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
    width: int 
    """Number of features in the autoencoder"""

    min_examples: int = 200
    """Minimum number of examples for a feature to be included"""

    max_examples: int = 10000
    """Maximum number of examples for a feature to included"""

    n_splits: int = 5
    """Number of splits that features were devided into"""


@dataclass
class CacheConfig(Serializable):

    dataset_repo: str = "EleutherAI/rpj-v2-sample"
    """Repository to load dataset from"""

    dataset_split: str = "train[:1%]"
    """Split of dataset to load"""

    dataset_name: str = ""
    """Name of dataset to load"""

    ctx_len: int = 64
    """Length of each example"""

    batch_size: int = 32
    """Number of sequences to process in a batch"""

    n_tokens: int = 10_000_000
    """Number of tokens to cache"""

    n_splits: int = 5
    """Number of splits to divide .safetensors into"""
