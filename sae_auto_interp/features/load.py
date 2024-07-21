import torch.multiprocessing as mp
from .features import Feature
from typing import List, Tuple, Dict

from .config import FeatureConfig
import torch

CONFIG = FeatureConfig()

def split_indices(width, n_splits, bounds=False):
    boundaries = torch.linspace(0, width, steps=n_splits+1).long()

    if bounds:
        return zip(boundaries[:-1], boundaries[1:])
    
    return boundaries

def load_selected(activations, locations, selected_features):

    def worker():
        pass

def _load(path, selected_features):

    activations = torch.load(path % "activations")
    locations = torch.load(path % "locations")

    return load_selected(activations, locations, selected_features)


    
def load(modules: List[str], raw_dir: str, cfg: FeatureConfig = CONFIG, features: Dict[str, int] = None):

    buckets = split_indices(cfg.width, cfg.n_splits)
    bounds = split_indices(cfg.width, cfg.n_splits, bounds=True)

    # pool = mp.Pool

    for module in modules:

        selected_features = features[module]

        bucketized = torch.bucketize(selected_features, buckets)
        unique_buckets = torch.unique(bucketized)

        for bucket in unique_buckets:
            mask = bucketized == bucket
            selected_features = selected_features[mask]

            start, end = bounds[bucket]

            path = f"{raw_dir}/{module}/{start}_{end}_%s.pt"

            _load(path)

