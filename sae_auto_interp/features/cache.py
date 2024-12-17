import json
import os
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from safetensors.numpy import save_file
from torchtyping import TensorType
from tqdm import tqdm

from sae_auto_interp.config import CacheConfig


class Cache:
    """
    The Cache class stores feature locations and activations for modules.
    It provides methods for adding, saving, and retrieving non-zero activations.
    """

    def __init__(
        self, filters: Dict[str, TensorType["indices"]] = None, batch_size: int = 64
    ):
        """
        Initialize the Cache.

        Args:
            filters (Dict[str, TensorType["indices"]], optional): Filters for selecting specific features.
            batch_size (int): Size of batches for processing. Defaults to 64.
        """
        self.feature_locations = defaultdict(list)
        self.feature_activations = defaultdict(list)
        self.filters = filters
        self.batch_size = batch_size

    def add(
        self,
        latents: TensorType["batch", "sequence", "feature"],
        batch_number: int,
        module_path: str,
    ):
        """
        Add the latents from a module to the cache.

        Args:
            latents (TensorType["batch", "sequence", "feature"]): Latent activations.
            batch_number (int): Current batch number.
            module_path (str): Path of the module.
        """
        feature_locations, feature_activations = self.get_nonzeros(latents, module_path)
        feature_locations = feature_locations.cpu()
        feature_activations = feature_activations.cpu()

        # Adjust batch indices
        feature_locations[:, 0] += batch_number * self.batch_size
        self.feature_locations[module_path].append(feature_locations)
        self.feature_activations[module_path].append(feature_activations)

    def save(self):
        """
        Concatenate the feature locations and activations for all modules.
        """
        for module_path in self.feature_locations.keys():
            self.feature_locations[module_path] = torch.cat(
                self.feature_locations[module_path], dim=0
            )

            self.feature_activations[module_path] = torch.cat(
                self.feature_activations[module_path], dim=0
            )

    def get_nonzeros_batch(self, latents: TensorType["batch", "seq", "feature"]):
        """
        Get non-zero activations for large batches that exceed int32 max value.

        Args:
            latents (TensorType["batch", "seq", "feature"]): Input latent activations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Non-zero feature locations and activations.
        """
        # Calculate the maximum batch size that fits within sys.maxsize
        max_batch_size = torch.iinfo(torch.int32).max // (latents.shape[1] * latents.shape[2])
        nonzero_feature_locations = []
        nonzero_feature_activations = []
        
        for i in range(0, latents.shape[0], max_batch_size):
            batch = latents[i:i+max_batch_size]
            
            # Get nonzero locations and activations
            batch_locations = torch.nonzero(batch.abs() > 1e-5)
            batch_activations = batch[batch.abs() > 1e-5]
            
            # Adjust indices to account for batching
            batch_locations[:, 0] += i 
            nonzero_feature_locations.append(batch_locations)
            nonzero_feature_activations.append(batch_activations)
        
        # Concatenate results
        nonzero_feature_locations = torch.cat(nonzero_feature_locations, dim=0)
        nonzero_feature_activations = torch.cat(nonzero_feature_activations, dim=0)
        return nonzero_feature_locations, nonzero_feature_activations

    def get_nonzeros(
        self, latents: TensorType["batch", "seq", "feature"], module_path: str
    ):
        """
        Get the nonzero feature locations and activations.

        Args:
            latents (TensorType["batch", "seq", "feature"]): Input latent activations.
            module_path (str): Path of the module.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Non-zero feature locations and activations.
        """
        size = latents.shape[1] * latents.shape[0] * latents.shape[2]
        if size > torch.iinfo(torch.int32).max:
            nonzero_feature_locations, nonzero_feature_activations = self.get_nonzeros_batch(latents)
        else:
            nonzero_feature_locations = torch.nonzero(latents.abs() > 1e-5)
            nonzero_feature_activations = latents[latents.abs() > 1e-5]
        # Return all nonzero features if no filter is provided
        if self.filters is None:
            return nonzero_feature_locations, nonzero_feature_activations

        # Return only the selected features if a filter is provided
        else:
            selected_features = self.filters[module_path]
            mask = torch.isin(nonzero_feature_locations[:, 2], selected_features)

            return nonzero_feature_locations[mask], nonzero_feature_activations[mask]


class FeatureCache:
    """
    FeatureCache manages the caching of feature activations for a model.
    It handles the process of running the model, storing activations, and saving them to disk.
    """

    def __init__(
        self,
        model,
        submodule_dict: Dict,
        batch_size: int,
        filters: Dict[str, TensorType["indices"]] = None,
    ):
        """
        Initialize the FeatureCache.

        Args:
            model: The model to cache features for.
            submodule_dict (Dict): Dictionary of submodules to cache.
            batch_size (int): Size of batches for processing.
            filters (Dict[str, TensorType["indices"]], optional): Filters for selecting specific features.
        """

        # Model must use FA2 to allow for efficient packing
        if not hasattr(model.config, "_attn_implementation") or model.config._attn_implementation != "flash_attention_2":
            raise ValueError("Model must use FlashAttention-2. Please enable it before initializing FeatureCache.")
        
        self.model = model
        self.submodule_dict = submodule_dict

        self.batch_size = batch_size
        self.width = list(submodule_dict.values())[0].ae.width

        self.cache = Cache(filters, batch_size=batch_size)
        if filters is not None:
            self.filter_submodules(filters)

        print(submodule_dict.keys())

    def load_token_batches(
        self, n_tokens: int, tokens: TensorType["batch", "sequence"]
    ):
        """
        Load and prepare token batches for processing.

        Args:
            n_tokens (int): Total number of tokens to process.
            tokens (TensorType["batch", "sequence"]): Input tokens.

        Returns:
            List[torch.Tensor]: List of token batches.
        """
        max_batches = n_tokens // tokens.shape[1]
        tokens = tokens[:max_batches]

        n_mini_batches = len(tokens) // self.batch_size

        token_batches = [
            tokens[self.batch_size * i : self.batch_size * (i + 1), :]
            for i in range(n_mini_batches)
        ]

        return token_batches

    def filter_submodules(self, filters: Dict[str, TensorType["indices"]]):
        """
        Filter submodules based on the provided filters.

        Args:
            filters (Dict[str, TensorType["indices"]]): Filters for selecting specific features.
        """
        filtered_submodules = {}
        for module_path in self.submodule_dict.keys():
            if module_path in filters:
                filtered_submodules[module_path] = self.submodule_dict[module_path]
        self.submodule_dict = filtered_submodules

    def run(self, n_tokens: int, tokens: TensorType["batch", "seq"]):
        """
        Run the feature caching process.

        Args:
            n_tokens (int): Total number of tokens to process.
            tokens (TensorType["batch", "seq"]): Input tokens.
        """
        token_batches = self.load_token_batches(n_tokens, tokens)

        total_tokens = 0
        total_batches = len(token_batches)
        tokens_per_batch = token_batches[0].numel()

        with tqdm(total=total_batches, desc="Caching features") as pbar:
            for batch_number, batch in enumerate(token_batches):
                total_tokens += tokens_per_batch

                with torch.no_grad():
                    buffer = {}
                    # position_ids is required for FA2
                    with self.model.trace({"input_ids": batch["input_ids"]}, position_ids=batch["position_ids"]):
                        for module_path, submodule in self.submodule_dict.items():
                            buffer[module_path] = submodule.ae.output.save()
                    for module_path, latents in buffer.items():
                        self.cache.add(latents, batch_number, module_path)

                    del buffer
                    torch.cuda.empty_cache()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"Total Tokens": f"{total_tokens:,}"})

        print(f"Total tokens processed: {total_tokens:,}")
        self.cache.save()

    def save(self, save_dir):
        """
        Save the cached features to disk.

        Args:
            save_dir (str): Directory to save the features.
        """
        for module_path in self.cache.feature_locations.keys():
            output_file = f"{save_dir}/{module_path}.safetensors"

            data = {
                "locations": self.cache.feature_locations[module_path],
                "activations": self.cache.feature_activations[module_path],
            }

            save_file(data, output_file)

    def _generate_split_indices(self, n_splits):
        """
        Generate indices for splitting the feature space.

        Args:
            n_splits (int): Number of splits to generate.

        Returns:
            List[Tuple[int, int]]: List of start and end indices for each split.
        """
        boundaries = torch.linspace(0, self.width, steps=n_splits + 1).long()

        # Adjust end by one
        return list(zip(boundaries[:-1], boundaries[1:] - 1))

    def save_splits(self, n_splits: int, save_dir):
        """
        Save the cached features in splits.

        Args:
            n_splits (int): Number of splits to generate.
            save_dir (str): Directory to save the splits.
        """
        split_indices = self._generate_split_indices(n_splits)

        for module_path in self.cache.feature_locations.keys():
            feature_locations = self.cache.feature_locations[module_path]
            feature_activations = self.cache.feature_activations[module_path]
            features = feature_locations[:, 2]

            for start, end in split_indices:
                
                mask = (features >= start) & (features <= end)

                masked_locations = feature_locations[mask].numpy()
                masked_activations = feature_activations[mask].half().numpy()
                masked_locations[:,2] = masked_locations[:,2]-start.item()
                if masked_locations[:,2].max() < 2**16 and masked_locations[:,0].max() < 2**16:
                    masked_locations = masked_locations.astype(np.uint16)
                else:
                    print(masked_locations[:,2].max(), masked_locations[:,0].max())
                    masked_locations = masked_locations.astype(np.uint32)
                
                module_dir = f"{save_dir}/{module_path}"
                os.makedirs(module_dir, exist_ok=True)

                output_file = f"{module_dir}/{start}_{end}.safetensors"

                split_data = {
                    "locations": masked_locations,
                    "activations": masked_activations,
                }

                save_file(split_data, output_file)

    def save_config(self, save_dir: str, cfg: CacheConfig, model_name: str):
        """
        Save the configuration for the cached features.

        Args:
            save_dir (str): Directory to save the configuration.
            cfg (CacheConfig): Configuration object.
            model_name (str): Name of the model.
        """
        for module_path in self.cache.feature_locations.keys():
            config_file = f"{save_dir}/{module_path}/config.json"
            with open(config_file, "w") as f:
                config_dict = cfg.to_dict()
                config_dict["model_name"] = model_name
                json.dump(config_dict, f)