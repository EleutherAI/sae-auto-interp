from simple_parsing import Serializable
from dataclasses import dataclass

@dataclass
class FeatureConfig(Serializable):
    
    width: int = 131_072

    n_splits: int = 2

@dataclass
class ExplainerConfig(Serializable):

    pass