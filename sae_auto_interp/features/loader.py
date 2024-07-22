import torch.multiprocessing as mp
from ..features.features import Feature, FeatureRecord
from typing import List, Tuple, Dict
from torchtyping import TensorType
from .transforms import Transform
from ..config import FeatureConfig
import time
import torch
from tqdm import tqdm


CONFIG = FeatureConfig()

class TensorBuffer:

    def __init__(self, module_path, features):
        self.module_path = module_path
        self.features = features
        self.n_features = len(features)
        
        self.start = 0

    def _load(self):

        self.activations = torch.load(self.module_path % "activations")
        self.locations = torch.load(self.module_path % "locations")

    def __iter__(self):
        self._load()

        return self
    
    def __next__(self):

        if self.start >= self.end:
            raise StopIteration
        
        feature = self.features[self.start]

        mask = self.locations[:, 2] == feature

        if mask.sum() == 0:
            self.start += 1
            return 0
        
        feature_locations = self.locations[mask][:,:2]
        feature_activations = self.activations[mask]

        self.start += 1

        return feature_locations, feature_activations

class FeatureDataset:

    def __init__(
        self, 
        raw_dir: str,
        modules: List[str],
        features: Dict[str, int],
        tokens: TensorType["batch", "sequence"], 
        constructor=None,
        sampler=None,
        transform: Transform = None,
        cfg: FeatureConfig = CONFIG,
    ):

        self.tokens = tokens
        self.constructor = constructor
        self.sampler = sampler
        self.transform = transform

        self.cfg = cfg
        self.buffers = self._load_buffers(
            raw_dir,
            modules, 
            features
        )

    def _load_buffers(self, raw_dir: str, modules: List[str], features: Dict[str, int]):
        
        edges = torch.linspace(
            0, 
            self.cfg.width, 
            steps=self.cfg.n_splits+1
        ).long()

        buffers = []

        for module in modules:

            selected_features = features[module]

            bucketized = torch.bucketize(selected_features, edges, right=True)
            unique_buckets = torch.unique(bucketized)

            for bucket in unique_buckets:
                mask = bucketized == bucket

                _selected_features = selected_features[mask]

                start, end = edges[bucket-1], edges[bucket]

                path = f"{raw_dir}/{module}/{start}_{end-1}_%s.pt"

                buffers.append(
                    TensorBuffer(
                        self.tokens, 
                        path, 
                        _selected_features
                    )
                )

        return buffers  

class FeatureLoader:

    def __init__(
        self,
        tokens,
        dataset: FeatureDataset,
        sampler=None,
        transform=None,
        constructor=None,
    ):
        self.dataset = dataset
        self.tokens = tokens
        self.constructor = constructor
        self.sampler = sampler
        self.transform = transform

    def _process(self, data):
        feature_locations, feature_activations = data

        record = FeatureRecord(self.feature)

        self.constructor(
            self.tokens,
            locations = feature_locations,
            activations = feature_activations,
        )

        if self.sampler is not None:
            self.sampler(
                record
            )

        if self.transform is not None:
            self.transform(
                record
            )

        return record
    
    def load(self, collate=False):
        all_records = []
        
        for buffer in self.dataset.buffers:
            buffer_records = [
                self._process(record)
                for record in tqdm(buffer)
                if record != 0
            ]
            
            if not collate:
                yield buffer_records
            else:
                all_records.extend(buffer_records)
        
        if collate:
            return all_records