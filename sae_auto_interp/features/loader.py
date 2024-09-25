import os
from typing import Callable, Dict, List, NamedTuple

import torch
from safetensors.numpy import load_file
import numpy as np
from torchtyping import TensorType
from tqdm import tqdm
import json

from sae_auto_interp.utils import (
    load_tokenized_data,
    load_tokenizer,
)

from ..config import FeatureConfig
from ..features.features import Feature, FeatureRecord

import asyncio
import time
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
        features: TensorType["features"],
        min_examples: int = 120,
    ):
        self.tensor_path = path
        self.module_path = module_path
        self.features = features
        self.min_examples = min_examples
        
        
    def __iter__(self):
        split_data = load_file(self.tensor_path)
        first_feature = int(self.tensor_path.split("/")[-1].split("_")[0])
        activations = torch.tensor(split_data["activations"])
        locations = torch.tensor(split_data["locations"].astype(np.int64))
        
        locations[:,2] = locations[:,2]+first_feature
        
        wanted_locations = torch.isin(locations[:,2],self.features)
        locations = locations[wanted_locations]
        activations = activations[wanted_locations]
        indices = torch.argsort(locations[:,2],stable=True)
        activations = activations[indices]
        locations = locations[indices]
        unique_features,counts = torch.unique_consecutive(locations[:,2],return_counts=True)
        features = unique_features
        split_locations = torch.split(locations,counts.tolist())
        split_activations = torch.split(activations,counts.tolist())
        
        for i in range(len(features)):
            
            feature_locations = split_locations[i]
            feature_activations = split_activations[i]
            if len(feature_locations) < self.min_examples:
                yield None
            else:
                yield BufferOutput(
                    Feature(self.module_path, features[i].item()),
                    feature_locations,
                    feature_activations
                )
        

    def reset(self):
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
        modules: list[str] = None,
        features: Dict[str, int] = None,
    ):
        self.cfg = cfg

        self.buffers = []

        if features is None:
            self._build(raw_dir, modules)

        else:
            self._build_selected(raw_dir, modules, features)

        cache_config_dir = f"{raw_dir}/{modules[0]}/config.json"
        with open(cache_config_dir, "r") as f:
            cache_config = json.load(f)
        self.tokenizer = load_tokenizer(cache_config["model_name"])
        self.tokens = load_tokenized_data(
            cache_config["ctx_len"],
            self.tokenizer,
            cache_config["dataset_repo"],
            cache_config["dataset_split"],
            cache_config["dataset_name"],
        )
   

    def _edges(self):
        return torch.linspace(0, self.cfg.width, steps=self.cfg.n_splits + 1).long()

    def _build(self, raw_dir: str, modules: list[str] = None):
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
        self, raw_dir: str, modules: list[str], features: Dict[str, int]
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
    
    def reset(self):
        for buffer in self.buffers:
            buffer.reset()

class FeatureLoader:
    def __init__(self, feature_dataset: 'FeatureDataset', constructor: Callable = None, sampler: Callable = None, transform: Callable = None):
        self.feature_dataset = feature_dataset
        self.constructor = constructor
        self.sampler = sampler
        self.transform = transform

    async def __aiter__(self):
        for buffer in self.feature_dataset.buffers:
            async for record in self._aprocess_buffer(buffer):
                yield record

    async def _aprocess_buffer(self, buffer):
        for data in buffer:
            if data is not None:
                #start_time = time.time()
                record = await self._aprocess_feature(data)
                #end_time = time.time()
                #print(f"Processed {data.feature} in {end_time - start_time} seconds")
                if record is not None:
                    yield record
            await asyncio.sleep(0)  # Allow other coroutines to run

    async def _aprocess_feature(self, buffer_output: BufferOutput):
        #start_time = time.time()
        #start_start = time.time()
        record = FeatureRecord(buffer_output.feature)
        #print(f"Feature record time: {time.time() - start_time} seconds")
        #print(buffer_output.feature)
        #start_time = time.time()
        if self.constructor is not None:
            self.constructor(record=record, buffer_output=buffer_output)
        #print(f"Constructor time: {time.time() - start_time} seconds")
        #start_time = time.time()
        if self.sampler is not None:
            self.sampler(record)
        #print(f"Sampler time: {time.time() - start_time} seconds")
        if self.transform is not None:
            self.transform(record)
        #print(f"it/s{1/(time.time() - start_start)}")
        return record

    def __iter__(self):
        for buffer in self.feature_dataset.buffers:
            start_start = time.time()
            for record in self._process_buffer(buffer):
                yield record
                #print(f"yielded it/s{1/(time.time() - start_start)}")
    def _process_buffer(self, buffer):
        start = time.time()
        for data in buffer:
            load_time = time.time()
            #print(f"load time: {load_time - start} seconds")
            start = time.time()
            if data is not None:
                #start_time = time.time()
                record = self._process_feature(data)
                #end_time = time.time()
                #print(f"Processed {data.feature} in {end_time - start_time} seconds")
                if record is not None:
                    yield record

    def _process_feature(self, buffer_output: BufferOutput):
        start_time = time.time()
        start_start = time.time()
        record = FeatureRecord(buffer_output.feature)
        #print(f"Feature record time: {time.time() - start_time} seconds")
        #print(buffer_output.feature)
        start_time = time.time()
        if self.constructor is not None:
            self.constructor(record=record, buffer_output=buffer_output)
        #print(f"Constructor time: {time.time() - start_time} seconds")
        start_time = time.time()
        if self.sampler is not None:
            self.sampler(record)
        #print(f"Sampler time: {time.time() - start_time} seconds")
        start_time = time.time()
        if self.transform is not None:
            self.transform(record)
        #print(f"it/s{1/(time.time() - start_start)}")
        return record