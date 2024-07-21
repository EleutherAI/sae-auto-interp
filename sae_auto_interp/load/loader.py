import torch.multiprocessing as mp
from ..features.features import Feature, FeatureRecord
from typing import List, Tuple, Dict

from ..features.config import FeatureConfig

import torch

import ray

CONFIG = FeatureConfig()

def split_indices(width, n_splits, bounds=False):
    boundaries = torch.linspace(0, width, steps=n_splits+1).long()

    if bounds:
        return zip(boundaries[:-1], boundaries[1:])
    
    return boundaries


@ray.remote
class Loader:

    def __init__(self, tokens, tokenizer, transforms):
        self.tokens = tokens
        self.tokenizer = tokenizer
        self.transforms = transforms

    def load_feature_batch(self, features, locations, activations):
        for feature in features:
            self.load_feature(
                feature,
                locations,
                activations
            )

    def load_feature(self, feature, locations, activations):
        mask = locations[:, 2] == feature
        feature_locations = locations[mask][:,:2]
        feature_activations = activations[mask]

        FeatureRecord.from_locations(
            Feature(0, feature), 
            self.tokens, 
            feature_locations, 
            feature_activations
        )

def load_module_shard(loader, path, selected_features, batch_size=20):

    activations = torch.load(path % "activations")
    locations = torch.load(path % "locations")

    feature_batches = torch.chunk(selected_features, batch_size)

    return [
        loader.load_feature_batch(
            batch,
            activations,
            locations
        ).remote()
        for batch in feature_batches
    ]

def load( 
    self,
    modules: List[str], 
    raw_dir: str, 
    cfg: FeatureConfig = CONFIG, 
    features: Dict[str, int] = None,
):

    buckets = split_indices(cfg.width, cfg.n_splits)
    bounds = split_indices(cfg.width, cfg.n_splits, bounds=True)

    for module in modules:

        selected_features = features[module]

        bucketized = torch.bucketize(selected_features, buckets)
        unique_buckets = torch.unique(bucketized)

        for bucket in unique_buckets:
            mask = bucketized == bucket
            selected_features = selected_features[mask]

            start, end = bounds[bucket]

            path = f"{raw_dir}/{module}/{start}_{end}_%s.pt"

            yield self.load_module(path, selected_features)