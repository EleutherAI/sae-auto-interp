import torch.multiprocessing as mp
from ..features.features import Feature, FeatureRecord
from typing import List, Tuple, Dict, NamedTuple
from torchtyping import TensorType
from .transforms import Transform
from ..config import FeatureConfig
import time
import torch
from tqdm import tqdm

CONFIG = FeatureConfig()


class BufferOutput(NamedTuple):
    feature: Feature

    locations: TensorType["locations", 2]

    activations: TensorType["locations"]

class TensorBuffer:
    """
    Buffer for loading tensors from disk. Returns an
    iterable that yields feature locations and activations.
    """

    def __init__(self, path, module_path, features):
        self.tensor_path = path
        self.module_path = module_path

        self.features = features
        self.start = 0

    def _load(self):

        self.activations = torch.load(self.tensor_path % "activations")
        self.locations = torch.load(self.tensor_path % "locations")

    def __iter__(self):
        self._load()

        if self.features is None:
            self.features = torch.unique(self.locations[:, 2])

        return self
    
    def __next__(self):

        if self.start >= len(self.features):
            raise StopIteration
        
        feature = self.features[self.start]

        mask = self.locations[:, 2] == feature

        # NOTE: MIN examples is here
        if mask.sum() <= 120:
            self.start += 1
            return None
        
        feature_locations = self.locations[mask][:,:2]
        feature_activations = self.activations[mask]

        self.start += 1

        return BufferOutput(
            Feature(
                self.module_path.replace("%s.pt_", ""),
                feature
            ),
            feature_locations,
            feature_activations
        )

class FeatureDataset:
    """
    Dataset which constructs TensorBuffers for each module and feature.
    """

    def __init__(
        self, 
        raw_dir: str,
        modules: List[str],
        features: Dict[str, int] = None,
        cfg: FeatureConfig = CONFIG,
    ):
        self.cfg = cfg

        self.buffers = []

        if features is None:

            self._load_all(raw_dir, modules)

        else:
        
            self._load_selected(
                raw_dir,
                modules, 
                features
            )

    def _edges(self):
        return torch.linspace(
            0, 
            self.cfg.width, 
            steps=self.cfg.n_splits+1
        ).long()
    

    def _load_all(self, raw_dir: str, modules: List[str]):
        edges = self._edges()

        for module in modules:
            for start, end in zip(edges[:-1], edges[1:]):

                # Adjust end by one as the path avoids overlap
                path = f"{raw_dir}/{module}/{start}_{end-1}_%s.pt"

                self.buffers.append(
                    TensorBuffer(
                        path, 
                        module,
                        None
                    )
                )

    def _load_selected(self, raw_dir: str, modules: List[str], features: Dict[str, int]):
        
        edges = self._edges()
        
        for module in modules:

            selected_features = features[module]

            bucketized = torch.bucketize(selected_features, edges, right=True)
            unique_buckets = torch.unique(bucketized)

            for bucket in unique_buckets:
                mask = bucketized == bucket

                _selected_features = selected_features[mask]

                start, end = edges[bucket-1], edges[bucket]

                # Adjust end by one as the path avoids overlap
                path = f"{raw_dir}/{module}/{start}_{end-1}_%s.pt"

                self.buffers.append(
                    TensorBuffer(
                        path, 
                        module,
                        _selected_features
                    )
                )

class FeatureLoader:
    """
    Loader which applies transformations and samplers to data.
    """

    def __init__(
        self,
        tokens: TensorType["batch", "seq"],
        dataset: FeatureDataset,
        sampler = None,
        transform = None,
        constructor = None,
    ):
        self.tokens = tokens
        self.dataset = dataset

        self.constructor = constructor
        self.sampler = sampler
        self.transform = transform

    def _process(self, data: BufferOutput):
        record = FeatureRecord(data.feature)

        self.constructor(
            record,
            self.tokens,
            locations = data.locations,
            activations = data.activations,
        )

        if self.sampler is not None:
            self.sampler(
                record
            )

        # if self.transform is not None:
        #     self.transform(
        #         record
        #     )

        return record
    
    def load_all(self):
        all_records = []

        for buffer in self.dataset.buffers:

            buffer_records = [
                self._process(data)
                for data in tqdm(buffer)
                if data is not None
            ]

            all_records.extend(buffer_records)

        return all_records
    
    def load(self):

        for buffer in self.dataset.buffers:

            buffer_records = [
                self._process(data)
                for data in tqdm(buffer)
                if data is not None
            ]

            yield buffer_records