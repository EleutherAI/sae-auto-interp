import yaml
import os
from dataclasses import dataclass, field

@dataclass
class CacheConfig:
    dataset_repo: str
    dataset_name: str
    dataset_split: str
    num_batches: int
    minibatch_size: int
    batch_len: int
    n_features: int
    n_tokens: int
    n_splits: int
    n_neighbors: int
    seed: int

@dataclass
class ExampleConfig:
    activation_threshold: float
    l: str
    r: str
    l_ctx: int
    r_ctx: int

@dataclass
class ChainOfThoughtExplainerConfig:
    temperature: float
    max_tokens: int

@dataclass
class SimpleExplainerConfig:
    temperature: float
    max_tokens: int

@dataclass
class DetectionScorerConfig:
    max_tokens: int
    temperature: float
    n_examples: int
    seed: int
    batch_size: int
    
@dataclass
class GenScorerConfig:
    n_tests: int
    temperature: float

with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as f:
    CONFIG = yaml.safe_load(f)

cache_cfg = CacheConfig(**CONFIG["cache"])
example_cfg = ExampleConfig(**CONFIG["example"])
cot_cfg = ChainOfThoughtExplainerConfig(**CONFIG["cot_explainer"])
simple_cfg = SimpleExplainerConfig(**CONFIG["simple_explainer"])
det_cfg = DetectionScorerConfig(**CONFIG["detection"])
gen_cfg = GenScorerConfig(**CONFIG["generation"])