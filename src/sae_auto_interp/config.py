from simple_parsing import Serializable
from dataclasses import dataclass

@dataclass
class FeatureConfig(Serializable):
    
    width: int = 131_072

    n_splits: int = 2

    n_train: int = 20

    n_test: int = 50

    n_quantiles: int = 10

@dataclass
class CacheConfig(Serializable):

    width: int = 131_072
    """Number of features in the autoencoder"""

    batch_size: int = 128
    """Number of sequences to process in a batch"""

    seq_len: int = 64
    """Context length of the autoencoder. Each batch is shape (batch_size, seq_len)"""

    n_tokens: int = 15_000_000
    """Number of tokens to cache"""

    n_splits: int = 4
    """Number of splits to divide .safetensors into"""