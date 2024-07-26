from simple_parsing import Serializable
from dataclasses import dataclass

@dataclass
class FeatureConfig(Serializable):
    
    width: int
    """Number of features in the autoencoder"""

    min_examples: int
    """Minimum number of examples for a feature to be included"""

    max_examples: int
    """Maximum number of examples for a feature to included"""

    ctx_len: int
    """Length of each example"""

    n_splits: int
    """Number of splits that features were devided into"""

    n_train: int 
    """Number of examples for generating an explanation"""

    n_test: int
    """Number of examples for evaluating an explanation, per quantile"""

    n_quantiles: int
    """Number of quantiles to draw test examples from"""

@dataclass
class CacheConfig(Serializable):

    batch_size: int
    """Number of sequences to process in a batch"""

    ctx_len: int
    """Context length of the autoencoder. Each batch is shape (batch_size, ctx_len)"""

    n_tokens: int
    """Number of tokens to cache"""

    n_splits: int
    """Number of splits to divide .safetensors into"""