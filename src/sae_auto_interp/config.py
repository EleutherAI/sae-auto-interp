from simple_parsing import Serializable
from dataclasses import dataclass

@dataclass
class FeatureConfig(Serializable):
    
    width: int = 131072
    """Number of features in the autoencoder"""

    min_examples: int = 200
    """Minimum number of activating examples for a feature to be included"""

    max_examples: int = 10000
    """Maximum number of activating examples to include for a feature"""

    ctx_len: int = 64
    """Length of each example"""

    n_splits: int = 5
    """Number of splits that features were devided into"""

    n_train: int = 10
    """Number of examples for generating an explanation"""

    n_test: int = 10
    """Number of examples for evaluating an explanation, per quantile"""

    n_quantiles: int = 5
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