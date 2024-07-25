from simple_parsing import Serializable
from dataclasses import dataclass

@dataclass
class FeatureConfig(Serializable):
    
    width: int = 131_072
    """Number of features in the autoencoder"""

    min_examples: int = 100
    """Minimum number of examples for a feature to be included"""

    max_examples: int = 5_000
    """Maximum number of examples for a feature to included"""

    ctx_len: int = 20
    """Length of each example"""

    n_splits: int = 2
    """Number of splits that features were devided into"""

    n_train: int = 20
    """Number of examples for generating an explanation"""

    n_test: int = 5
    """Number of examples for evaluating an explanation, per quantile"""

    n_quantiles: int = 10
    """Number of quantiles to draw test examples from"""

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

    n_splits: int = 2
    """Number of splits to divide .safetensors into"""