import yaml
import os
from dataclasses import dataclass, field

CONFIG_PATH = os.environ.get("CONFIG_PATH", None)

@dataclass
class CacheConfig:
    dataset_repo: str
    dataset_split: str
    dataset_name: str
    minibatch_size: int
    seq_len: int
    n_features: int
    n_tokens: int
    seed: int

    l_ctx: int = 0
    r_ctx: int = 0

@dataclass
class ChainOfThoughtExplainerConfig:
    temperature: float
    max_tokens: int
    l: str
    r: str
    threshold: float

@dataclass
class SimpleExplainerConfig:
    temperature: float
    max_tokens: int

@dataclass
class DetectionScorerConfig:
    max_tokens: int
    temperature: float
    seed: int
    batch_size: int
    threshold: float

    l: str
    r: str
    
@dataclass
class GenScorerConfig:
    n_tests: int
    temperature: float

with open(os.path.join(os.path.dirname(__file__), CONFIG_PATH), 'r') as f:
    CONFIG = yaml.safe_load(f)

cache_config = CacheConfig(**CONFIG["cache"])
cot_explainer_config = ChainOfThoughtExplainerConfig(**CONFIG["cot_explainer"])
simple_explainer_config = SimpleExplainerConfig(**CONFIG["simple_explainer"])
det_config = DetectionScorerConfig(**CONFIG["detection"])
gen_config = GenScorerConfig(**CONFIG["generation"])
log_path = CONFIG["log_path"]