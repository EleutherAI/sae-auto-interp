import asyncio
import json
import os
from typing import Callable, Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch
from nnsight import LanguageModel
from safetensors.numpy import load_file
from torchtyping import TensorType

from delphi.utils import (
    load_tokenized_data,
)

from ..config import LatentConfig
from .latents import Latent, LatentRecord


class AllData(NamedTuple):
    features: list[TensorType["features"]]
    locations: list[TensorType["locations", 2]]
    activations: list[TensorType["locations"]]
    tokens: TensorType["tokens"]


class BufferOutput(NamedTuple):
    """
    Represents the output of a TensorBuffer.

    Attributes:
        latent (Latent): The latent associated with this output.
        locations (TensorType["locations", 2]): Tensor of latent locations.
        activations (TensorType["locations"]): Tensor of latent activations.
        tokens (TensorType["tokens"]): Tensor of all tokens.
    """
    latent: Latent
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
        latents: Optional[TensorType["latents"]] = None,
        min_examples: int = 120,
    ):
        """
        Initialize a TensorBuffer.

        Args:
            path (str): Path to the tensor file.
            module_path (str): Path of the module.
            latents (Optional[TensorType["latents"]]): Tensor of latent indices.
            min_examples (int): Minimum number of examples required. Defaults to 120.
        """
        self.tensor_path = path
        self.module_path = module_path
        self.latents = latents
        self.min_examples = min_examples
        
   

    def __iter__(self):
        """
        Iterate over the buffer, yielding BufferOutput objects.

        Yields:
            Union[BufferOutput, None]: BufferOutput if enough examples, None otherwise.
        """
        latents, split_locations, split_activations, tokens = self.load()
        
        for i in range(len(latents)):
            latent_locations = split_locations[i]
            latent_activations = split_activations[i]
            if len(latent_locations) < self.min_examples:
                yield None
            else:
                yield BufferOutput(
                    Latent(self.module_path, int(latents[i].item())),
                    latent_locations,
                    latent_activations,
                )

    def load(self):
        split_data = load_file(self.tensor_path)
        first_latent = int(self.tensor_path.split("/")[-1].split("_")[0])
        activations = torch.tensor(split_data["activations"])
        locations = torch.tensor(split_data["locations"].astype(np.int64))
        if "tokens" in split_data:
            tokens = torch.tensor(split_data["tokens"].astype(np.int64))
        else:
            tokens = None
        
        locations[:,2] = locations[:,2] + first_latent
        
        if self.latents is not None:
            wanted_locations = torch.isin(locations[:,2], self.latents)
            locations = locations[wanted_locations]
            activations = activations[wanted_locations]
        
        indices = torch.argsort(locations[:,2], stable=True)
        activations = activations[indices]
        locations = locations[indices]
        unique_latents, counts = torch.unique_consecutive(locations[:,2], return_counts=True)
        latents = unique_latents
        split_locations = torch.split(locations, counts.tolist())
        split_activations = torch.split(activations, counts.tolist())

        return latents, split_locations, split_activations, tokens


    def reset(self):
        """Reset the buffer state."""
        self.start = 0
        self.activations = None
        self.locations = None


class LatentDataset:
    """
    Dataset which constructs TensorBuffers for each module and latent.
    """

    def __init__(
        self,
        raw_dir: str,
        cfg: LatentConfig,
        tokenizer: Optional[Callable] = None,
        modules: Optional[List[str]] = None,
        latents: Optional[Dict[str, Union[int, torch.Tensor]]] = None,
    ):
        """
        Initialize a LatentDataset.

        Args:
            raw_dir (str): Directory containing raw latent data.
            cfg (LatentConfig): Configuration for latent processing.
            modules (Optional[List[str]]): List of module names to include.
            latents (Optional[Dict[str, Union[int, torch.Tensor]]]): Dictionary of latents per module.
        """
        self.cfg = cfg
        self.buffers = []
        

        if latents is None:
            self._build(raw_dir, modules)
        else:
            # TODO fix type error
            self._build_selected(raw_dir, modules, latents) # type: ignore

        self.all_data = self._build_everything(raw_dir, modules)

        cache_config_dir = f"{raw_dir}/{modules[0]}/config.json"
        with open(cache_config_dir, "r") as f:
            cache_config = json.load(f)
        if tokenizer is None:
            temp_model = LanguageModel(cache_config["model_name"], device_map="cpu", dispatch=False)
            self.tokenizer = temp_model.tokenizer
        else:
            self.tokenizer = tokenizer
        self.cache_config = cache_config

    def load_tokens(self):
        """
        Load tokenized data for the dataset.
        Caches the tokenized data if not already loaded.
        
        Returns:
            torch.Tensor: The tokenized dataset.
        """
        if not hasattr(self, "tokens"):
            self.tokens = load_tokenized_data(
                self.cache_config["ctx_len"],
                self.tokenizer,
                self.cache_config["dataset_repo"],
                self.cache_config["dataset_split"],
                self.cache_config["dataset_name"],
                column_name=self.cache_config.get(
                    "dataset_column_name", self.cache_config.get("dataset_row", "raw_content")
                ),
            )
        return self.tokens

    def _edges(self):
        """Generate edge indices for latent splits."""
        return torch.linspace(0, self.cfg.width, steps=self.cfg.n_splits + 1).long()

    def _build(self, raw_dir: str, modules: Optional[List[str]] = None):
        """
        Build dataset buffers which load all cached latents.

        Args:
            raw_dir (str): Directory containing raw latent data.
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
        self, raw_dir: str, modules: List[str], latents: Dict[str, Union[int, torch.Tensor]]
    ):
        """
        Build a dataset buffer which loads only selected latents.

        Args:
            raw_dir (str): Directory containing raw latent data.
            modules (List[str]): List of module names to include.
            latents (Dict[str, Union[int, torch.Tensor]]): Dictionary of latents per module.
        """
        edges = self._edges()

        for module in modules:
            selected_latents = latents[module]
            if isinstance(selected_latents, int):
                selected_latents = torch.tensor([selected_latents])
            
            bucketized = torch.bucketize(selected_latents, edges, right=True)
            unique_buckets = torch.unique(bucketized)

            for bucket in unique_buckets:
                mask = bucketized == bucket
                _selected_latents = selected_latents[mask]

                start, end = edges[bucket.item() - 1], edges[bucket.item()]

                # Adjust end by one as the path avoids overlap
                path = f"{raw_dir}/{module}/{start}_{end-1}.safetensors"
                
                self.buffers.append(
                    TensorBuffer(
                        path,
                        module,
                        _selected_latents,
                        min_examples=self.cfg.min_examples,
                    )
                )

    def _build_everything(self, raw_dir: str, modules: List[str]):
        """
        Build a BufferOutput with the locations and activations of all features.
        """
        edges = self._edges()
        all_features = []
        all_locations = []
        all_activations = []
        tokens = None
        all_data = {}
        for module in modules:
            for start, end in zip(edges[:-1], edges[1:]):
                path = f"{raw_dir}/{module}/{start}_{end-1}.safetensors"
                
                buffer = TensorBuffer(path, module, min_examples=self.cfg.min_examples)
                features, locations, activations, tk = buffer.load()
                all_features.extend(features)
                all_locations.extend(locations)
                all_activations.extend(activations)
                if tokens is None:
                    tokens = tk
            all_features = torch.stack(all_features)
            all_data[module] = AllData(all_features, all_locations, all_activations, tokens)


        return all_data

    def __len__(self):
        """Return the number of buffers in the dataset."""
        return len(self.buffers)
        
    def reset(self):
        """Reset all buffers in the dataset."""
        for buffer in self.buffers:
            buffer.reset()

class LatentLoader:
    """
    Loader class for processing latent records from a LatentDataset.
    """

    def __init__(
        self,
        latent_dataset: 'LatentDataset',
        constructor: Optional[Callable] = None,
        sampler: Optional[Callable] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize a LatentLoader.

        Args:
            latent_dataset (LatentDataset): The dataset to load latents from.
            constructor (Optional[Callable]): Function to construct latent records.
            sampler (Optional[Callable]): Function to sample from latent records.
            transform (Optional[Callable]): Function to transform latent records.
        """
        self.latent_dataset = latent_dataset
        self.constructor = constructor
        self.sampler = sampler
        self.transform = transform

    async def __aiter__(self):
        """
        Asynchronous iterator for processing latent records.

        Yields:
            LatentRecord: Processed latent records.
        """
        for buffer in self.latent_dataset.buffers:
            for data in buffer:
                if data is not None:
                    
                    record = await self._aprocess_latent(data)
                    if record is not None:
                        print()
                        yield record
            await asyncio.sleep(0)

    async def _aprocess_latent(self, buffer_output: BufferOutput):
        """
        Asynchronously process a single latent.

        Args:
            buffer_output (BufferOutput): Latent data to process.

        Returns:
            Optional[LatentRecord]: Processed latent record or None.
        """
        record = LatentRecord(buffer_output.latent)
        if self.transform is not None:
            self.transform(record)
        if self.constructor is not None:
            self.constructor(record=record, buffer_output=buffer_output)
        if self.sampler is not None:
            self.sampler(record)
        return record
