import os
from functools import partial, reduce
from typing import Callable, Dict, List, NamedTuple

import torch
import torch.multiprocessing as mp
from safetensors.torch import load_file
from torchtyping import TensorType
from tqdm import tqdm

from ..config import FeatureConfig
from ..features.features import Feature, FeatureRecord


class BufferOutput(NamedTuple):
    feature: Feature

    locations: TensorType["locations", 2]

    activations: TensorType["locations"]


class TensorBuffer:
    """
    Lazy loading buffer for cached splits.
    """

    def __init__(
        self,
        path: str,
        module_path: str,
        features: TensorType["features"] = None,
        min_examples: int = 120,
    ):
        self.tensor_path = path
        self.module_path = module_path

        self.features = features
        self.min_examples = min_examples
        self.start = 0

        self.activations = None
        self.locations = None

    def _load(self):
        split_data = load_file(self.tensor_path)

        self.activations = split_data["activations"]
        self.locations = split_data["locations"]

    def __iter__(self):
        self._load()

        if self.features is None:
            self.features = torch.unique(self.locations[:, 2])

        return self

    def __next__(self):
        if self.start >= len(self.features):
            del self.activations
            del self.locations
            torch.cuda.empty_cache()

            raise StopIteration

        feature = self.features[self.start]

        mask = self.locations[:, 2] == feature

        # NOTE: MIN examples is here
        if mask.sum() < self.min_examples:
            self.start += 1
            return None

        feature_locations = self.locations[mask][:, :2]
        feature_activations = self.activations[mask]

        self.start += 1

        return BufferOutput(
            Feature(self.module_path, feature.item()),
            feature_locations,
            feature_activations,
        )


class FeatureDataset:
    """
    Dataset which constructs TensorBuffers for each module and feature.
    """

    def __init__(
        self,
        raw_dir: str,
        cfg: FeatureConfig,
        modules: List[str] = None,
        features: Dict[str, int] = None,
    ):
        self.cfg = cfg

        self.buffers = []

        if features is None:
            self._build(raw_dir, modules)

        else:
            self._build_selected(raw_dir, modules, features)

    def _edges(self):
        return torch.linspace(0, self.cfg.width, steps=self.cfg.n_splits + 1).long()

    def _build(self, raw_dir: str, modules: List[str] = None):
        """
        Build dataset buffers which load all cached features.
        """

        edges = self._edges()

        modules = os.listdir(raw_dir) if modules is None else modules

        for module in modules:
            for start, end in zip(edges[:-1], edges[1:]):
                # Adjust end by one as the path avoids overlap
                path = f"{raw_dir}/{module}/{start}_{end-1}.safetensors"

                self.buffers.append(
                    TensorBuffer(path, module, min_examples=self.cfg.min_examples)
                )

    def _build_selected(
        self, raw_dir: str, modules: List[str], features: Dict[str, int]
    ):
        """
        Build a dataset buffer which loads only selected features.
        """

        edges = self._edges()

        for module in modules:
            selected_features = features[module]

            bucketized = torch.bucketize(selected_features, edges, right=True)
            unique_buckets = torch.unique(bucketized)

            for bucket in unique_buckets:
                mask = bucketized == bucket

                _selected_features = selected_features[mask]

                start, end = edges[bucket - 1], edges[bucket]

                # Adjust end by one as the path avoids overlap
                path = f"{raw_dir}/{module}/{start}_{end-1}.safetensors"

                self.buffers.append(
                    TensorBuffer(
                        path,
                        module,
                        _selected_features,
                        min_examples=self.cfg.min_examples,
                    )
                )

    def __len__(self):
        return len(self.buffers)
        
    def load(
        self,
        collate: bool = False,
        constructor: Callable = None,
        sampler: Callable = None,
        transform: Callable = None,
    ):
        def _process(buffer_output: BufferOutput):
            record = FeatureRecord(buffer_output.feature)
            if constructor is not None:
                constructor(record=record, buffer_output=buffer_output)

            if sampler is not None:
                sampler(record)

            if transform is not None:
                transform(record)

            return record

        def _worker(buffer):
            return [
                _process(data)
                for data in tqdm(buffer, desc=f"Loading {buffer.module_path}")
                if data is not None
            ]

        return self._load(collate, _worker)
    
    def _load(self, collate: bool, _worker: Callable):
        if collate:
            all_records = []
            for buffer in self.buffers:
                all_records.extend(_worker(buffer))
            return all_records

        else:
            for buffer in self.buffers:
                yield _worker(buffer)
