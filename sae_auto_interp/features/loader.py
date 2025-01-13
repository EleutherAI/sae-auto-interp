import asyncio
import json
import os
from typing import Callable, Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch
from safetensors.numpy import load_file
from torchtyping import TensorType
from tqdm import tqdm
from nnsight import LanguageModel

from sae_auto_interp.utils import (
    load_tokenized_data,
    load_tokenizer,
)

from ..config import FeatureConfig
from ..features.features import Feature, FeatureRecord


class BufferOutput(NamedTuple):
    """
    Represents the output of a TensorBuffer.

    Attributes:
        feature (Feature): The feature associated with this output.
        locations (TensorType["locations", 2]): Tensor of feature locations.
        activations (TensorType["locations"]): Tensor of feature activations.
    """
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
        features: Optional[TensorType["features"]] = None,
        min_examples: int = 120,
    ):
        """
        Initialize a TensorBuffer.

        Args:
            path (str): Path to the tensor file.
            module_path (str): Path of the module.
            features (Optional[TensorType["features"]]): Tensor of feature indices.
            min_examples (int): Minimum number of examples required. Defaults to 120.
        """
        self.tensor_path = path
        self.module_path = module_path
        self.features = features
        self.min_examples = min_examples
        
    def __iter__(self):
        """
        Iterate over the buffer, yielding BufferOutput objects.

        Yields:
            Union[BufferOutput, None]: BufferOutput if enough examples, None otherwise.
        """
        split_data = load_file(self.tensor_path)
        first_feature = int(self.tensor_path.split("/")[-1].split("_")[0])
        activations = torch.tensor(split_data["activations"])
        locations = torch.tensor(split_data["locations"].astype(np.int64))
        
        locations[:,2] = locations[:,2] + first_feature
        
        if self.features is not None:
            wanted_locations = torch.isin(locations[:,2], self.features)
            locations = locations[wanted_locations]
            activations = activations[wanted_locations]
        
        indices = torch.argsort(locations[:,2], stable=True)
        activations = activations[indices]
        locations = locations[indices]
        unique_features, counts = torch.unique_consecutive(locations[:,2], return_counts=True)
        features = unique_features
        split_locations = torch.split(locations, counts.tolist())
        split_activations = torch.split(activations, counts.tolist())
        
        for i in range(len(features)):
            feature_locations = split_locations[i]
            feature_activations = split_activations[i]
            if len(feature_locations) < self.min_examples:
                yield None
            else:
                yield BufferOutput(
                    Feature(self.module_path, int(features[i].item())),
                    feature_locations,
                    feature_activations
                )

    def reset(self):
        """Reset the buffer state."""
        self.start = 0
        self.activations = None
        self.locations = None


class FeatureDataset:
    """
    Dataset which constructs TensorBuffers for each module and feature.
    """

    def __init__(
        self,
        raw_dir: str,
        cfg: FeatureConfig,
        modules: Optional[List[str]] = None,
        features: Optional[Dict[str, Union[int, torch.Tensor]]] = None,
    ):
        """
        Initialize a FeatureDataset.

        Args:
            raw_dir (str): Directory containing raw feature data.
            cfg (FeatureConfig): Configuration for feature processing.
            modules (Optional[List[str]]): List of module names to include.
            features (Optional[Dict[str, Union[int, torch.Tensor]]]): Dictionary of features per module.
        """
        self.cfg = cfg
        self.buffers = []

        if features is None:
            self._build(raw_dir, modules)
        else:
            self._build_selected(raw_dir, modules, features)

        cache_config_dir = f"{raw_dir}/{modules[0]}/config.json"
        with open(cache_config_dir, "r") as f:
            cache_config = json.load(f)
        temp_model = LanguageModel(cache_config["model_name"], device_map="cpu", dispatch=False)
        self.tokenizer = temp_model.tokenizer
        print(cache_config)
        self.tokens = load_tokenized_data(
            cache_config["ctx_len"],
            self.tokenizer,
            cache_config["dataset_repo"],
            cache_config["dataset_split"],
            cache_config["dataset_name"],
            cache_config["dataset_column_name"],
        )
        print(self.tokenizer.decode(self.tokens[0]))
   
    def _edges(self):
        """Generate edge indices for feature splits."""
        return torch.linspace(0, self.cfg.width, steps=self.cfg.n_splits + 1).long()

    def _build(self, raw_dir: str, modules: Optional[List[str]] = None):
        """
        Build dataset buffers which load all cached features.

        Args:
            raw_dir (str): Directory containing raw feature data.
            modules (Optional[List[str]]): List of module names to include.
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
        self, raw_dir: str, modules: List[str], features: Dict[str, Union[int, torch.Tensor]]
    ):
        """
        Build a dataset buffer which loads only selected features.

        Args:
            raw_dir (str): Directory containing raw feature data.
            modules (List[str]): List of module names to include.
            features (Dict[str, Union[int, torch.Tensor]]): Dictionary of features per module.
        """
        edges = self._edges()

        for module in modules:
            selected_features = features[module]
            if isinstance(selected_features, int):
                selected_features = torch.tensor([selected_features])
            
            bucketized = torch.bucketize(selected_features, edges, right=True)
            unique_buckets = torch.unique(bucketized)

            for bucket in unique_buckets:
                mask = bucketized == bucket
                _selected_features = selected_features[mask]

                start, end = edges[bucket.item() - 1], edges[bucket.item()]

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
        """Return the number of buffers in the dataset."""
        return len(self.buffers)
        
    def load(
        self,
        collate: bool = False,
        constructor: Optional[Callable] = None,
        sampler: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Load and process feature records from the dataset.

        Args:
            collate (bool): Whether to collate all records into a single list.
            constructor (Optional[Callable]): Function to construct feature records.
            sampler (Optional[Callable]): Function to sample from feature records.
            transform (Optional[Callable]): Function to transform feature records.

        Returns:
            Union[List[FeatureRecord], Generator]: Processed feature records.
        """
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
        """
        Internal method to load feature records.

        Args:
            collate (bool): Whether to collate all records into a single list.
            _worker (Callable): Function to process each buffer.

        Returns:
            Union[List[FeatureRecord], Generator]: Processed feature records.
        """
        if collate:
            all_records = []
            for buffer in self.buffers:
                all_records.extend(_worker(buffer))
            return all_records
        else:
            for buffer in self.buffers:
                yield _worker(buffer)
    
    def reset(self):
        """Reset all buffers in the dataset."""
        for buffer in self.buffers:
            buffer.reset()

class FeatureLoader:
    """
    Loader class for processing feature records from a FeatureDataset.
    """

    def __init__(
        self,
        feature_dataset: 'FeatureDataset',
        constructor: Optional[Callable] = None,
        sampler: Optional[Callable] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize a FeatureLoader.

        Args:
            feature_dataset (FeatureDataset): The dataset to load features from.
            constructor (Optional[Callable]): Function to construct feature records.
            sampler (Optional[Callable]): Function to sample from feature records.
            transform (Optional[Callable]): Function to transform feature records.
        """
        self.feature_dataset = feature_dataset
        self.constructor = constructor
        self.sampler = sampler
        self.transform = transform

    async def __aiter__(self):
        """
        Asynchronous iterator for processing feature records.

        Yields:
            FeatureRecord: Processed feature records.
        """
        for buffer in self.feature_dataset.buffers:
            async for record in self._aprocess_buffer(buffer):
                yield record

    async def _aprocess_buffer(self, buffer):
        """
        Asynchronously process a buffer.

        Args:
            buffer (TensorBuffer): Buffer to process.

        Yields:
            Optional[FeatureRecord]: Processed feature record or None.
        """
        for data in buffer:
            if data is not None:
                record = await self._aprocess_feature(data)
                if record is not None:
                    yield record
            await asyncio.sleep(0)  # Allow other coroutines to run

    async def _aprocess_feature(self, buffer_output: BufferOutput):
        """
        Asynchronously process a single feature.

        Args:
            buffer_output (BufferOutput): Feature data to process.

        Returns:
            Optional[FeatureRecord]: Processed feature record or None.
        """
        record = FeatureRecord(buffer_output.feature)
        if self.constructor is not None:
            self.constructor(record=record, buffer_output=buffer_output)
        if self.sampler is not None:
            self.sampler(record)
        if self.transform is not None:
            self.transform(record)
        return record

    def __iter__(self):
        """
        Synchronous iterator for processing feature records.

        Yields:
            FeatureRecord: Processed feature records.
        """
        for buffer in self.feature_dataset.buffers:
            for record in self._process_buffer(buffer):
                yield record

    def _process_buffer(self, buffer):
        """
        Process a buffer synchronously.

        Args:
            buffer (TensorBuffer): Buffer to process.

        Yields:
            Optional[FeatureRecord]: Processed feature record or None.
        """
        for data in buffer:
            if data is not None:
                record = self._process_feature(data)
                if record is not None:
                    yield record

    def _process_feature(self, buffer_output: BufferOutput):
        """
        Process a single feature synchronously.

        Args:
            buffer_output (BufferOutput): Feature data to process.

        Returns:
            Optional[FeatureRecord]: Processed feature record or None.
        """
        record = FeatureRecord(buffer_output.feature)
        if self.constructor is not None:
            self.constructor(record=record, buffer_output=buffer_output)
        if self.sampler is not None:
            self.sampler(record)
        if self.transform is not None:
            self.transform(record)
        return record