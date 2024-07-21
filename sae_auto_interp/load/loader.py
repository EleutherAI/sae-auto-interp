import torch.multiprocessing as mp
from ..features.features import Feature, FeatureRecord
from typing import List, Tuple, Dict
from torchtyping import TensorType
from .transforms import Transform
from ..features.config import FeatureConfig
import time
import torch


CONFIG = FeatureConfig()


class TensorBuffer:

    def __init__(self, tokens, module_path, features, start=0, end=None):
        self.tokens = tokens
        self.module_path = module_path
        self.features = features
        self.n_features = len(features)
        
        self.start = start
        self.end = self.n_features if end is None else end

    def load(self):

        self.activations = torch.load(self.module_path % "activations")
        self.locations = torch.load(self.module_path % "locations")

        self.activations.share_memory_()
        self.locations.share_memory_()

    def __iter__(self):
        return self
    
    def set_edges(self, start: int, end: int):
        self.start = start
        self.end = end

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

        record = FeatureRecord.load(
            Feature(self.module_path, feature), 
            self.tokens, 
            feature_locations,
            feature_activations
        )

        self.start += 1
        return record

class FeatureDataset:

    def __init__(
        self, 
        raw_dir: str,
        modules: List[str],
        features: Dict[str, int],
        tokens: TensorType["batch", "sequence"], 
        sampler=None,
        transform: Transform = None,
        cfg: FeatureConfig = CONFIG,
    ):

        tokens.share_memory_()
        self.tokens = tokens
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
    
    def _load_worker(self, q, buffer: TensorBuffer, start: int, end: int):
        buffer.set_edges(start, end)
        records = []
        for record in buffer:
            records.append(record)

        q.put(records)

        

    def load(self, n_workers: int = 1):

        for buffer in self.buffers:
            with mp.Manager() as manager:
                processes = []
                q = manager.Queue()
                buffer.load()

                bounds = torch.linspace(
                    0, 
                    buffer.n_features, 
                    steps=n_workers+1,
                ).long()

                for i in range(n_workers):
                    start, end = bounds[i], bounds[i+1]

                    process = mp.Process(
                        target=self._load_worker, 
                        args=(q, buffer, start, end)
                    )

                    processes.append(process)
                    process.start()

                results = []
                for process in processes:
                    process.join()
                    results.append(q.get())

            yield results