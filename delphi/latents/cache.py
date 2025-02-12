import json
import os
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from safetensors.numpy import save_file
from torchtyping import TensorType
from tqdm import tqdm

from delphi.config import CacheConfig


class Cache:
    """
    The Cache class stores latent locations and activations for modules.
    It provides methods for adding, saving, and retrieving non-zero activations.
    """

    def __init__(
        self, filters: Dict[str, TensorType["indices"]] = None, batch_size: int = 64
    ):
        """
        Initialize the Cache.

        Args:
            filters (Dict[str, TensorType["indices"]], optional): Filters for selecting specific latents.
            batch_size (int): Size of batches for processing. Defaults to 64.
        """
        self.latent_locations = defaultdict(list)
        self.latent_activations = defaultdict(list)
        self.tokens = defaultdict(list)
        self.filters = filters
        self.batch_size = batch_size

    def add(
        self,
        latents: TensorType["batch", "sequence", "latent"],
        tokens: TensorType["batch", "sequence"],
        batch_number: int,
        module_path: str,
    ):
        """
        Add the latents from a module to the cache.

        Args:
            latents (TensorType["batch", "sequence", "latent"]): Latent activations.
            tokens (TensorType["batch", "sequence"]): Input tokens.
            batch_number (int): Current batch number.
            module_path (str): Path of the module.
        """
        latent_locations, latent_activations = self.get_nonzeros(latents, module_path)
        latent_locations = latent_locations.cpu()
        latent_activations = latent_activations.cpu()
        tokens = tokens.cpu()

        # Adjust batch indices
        latent_locations[:, 0] += batch_number * self.batch_size
        self.latent_locations[module_path].append(latent_locations)
        self.latent_activations[module_path].append(latent_activations)
        self.tokens[module_path].append(tokens)

    def save(self):
        """
        Concatenate the latent locations and activations for all modules.
        """
        for module_path in self.latent_locations.keys():
            self.latent_locations[module_path] = torch.cat(
                self.latent_locations[module_path], dim=0
            )

            self.latent_activations[module_path] = torch.cat(
                self.latent_activations[module_path], dim=0
            )

            self.tokens[module_path] = torch.cat(self.tokens[module_path], dim=0)

    def get_nonzeros_batch(self, latents: TensorType["batch", "seq", "latent"]):
        """
        Get non-zero activations for large batches that exceed int32 max value.

        Args:
            latents (TensorType["batch", "seq", "latent"]): Input latent activations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Non-zero latent locations and activations.
        """
        # Calculate the maximum batch size that fits within sys.maxsize
        max_batch_size = torch.iinfo(torch.int32).max // (
            latents.shape[1] * latents.shape[2]
        )
        nonzero_latent_locations = []
        nonzero_latent_activations = []

        for i in range(0, latents.shape[0], max_batch_size):
            batch = latents[i : i + max_batch_size]

            # Get nonzero locations and activations
            batch_locations = torch.nonzero(batch.abs() > 1e-5)
            batch_activations = batch[batch.abs() > 1e-5]

            # Adjust indices to account for batching
            batch_locations[:, 0] += i
            nonzero_latent_locations.append(batch_locations)
            nonzero_latent_activations.append(batch_activations)

        # Concatenate results
        nonzero_latent_locations = torch.cat(nonzero_latent_locations, dim=0)
        nonzero_latent_activations = torch.cat(nonzero_latent_activations, dim=0)
        return nonzero_latent_locations, nonzero_latent_activations

    def get_nonzeros(
        self, latents: TensorType["batch", "seq", "latent"], module_path: str
    ):
        """
        Get the nonzero latent locations and activations.

        Args:
            latents (TensorType["batch", "seq", "latent"]): Input latent activations.
            module_path (str): Path of the module.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Non-zero latent locations and activations.
        """
        size = latents.shape[1] * latents.shape[0] * latents.shape[2]
        if size > torch.iinfo(torch.int32).max:
            (
                nonzero_latent_locations,
                nonzero_latent_activations,
            ) = self.get_nonzeros_batch(latents)
        else:
            nonzero_latent_locations = torch.nonzero(latents.abs() > 1e-5)
            nonzero_latent_activations = latents[latents.abs() > 1e-5]

        # Return all nonzero latents if no filter is provided
        if self.filters is None:
            return nonzero_latent_locations, nonzero_latent_activations

        # Return only the selected latents if a filter is provided
        else:
            selected_latents = self.filters[module_path]
            mask = torch.isin(nonzero_latent_locations[:, 2], selected_latents)

            return nonzero_latent_locations[mask], nonzero_latent_activations[mask]


class LatentCache:
    """
    LatentCache manages the caching of latent activations for a model.
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
        Initialize the LatentCache.

        Args:
            model: The model to cache latents for.
            submodule_dict (Dict): Dictionary of submodules to cache.
            batch_size (int): Size of batches for processing.
            filters (Dict[str, TensorType["indices"]], optional): Filters for selecting specific latents.
        """
        self.model = model
        self.submodule_dict = submodule_dict

        self.batch_size = batch_size
        self.width = list(submodule_dict.values())[0].ae.width

        self.cache = Cache(filters, batch_size=batch_size)
        if filters is not None:
            self.filter_submodules(filters)

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
            filters (Dict[str, TensorType["indices"]]): Filters for selecting specific latents.
        """
        filtered_submodules = {}
        for module_path in self.submodule_dict.keys():
            if module_path in filters:
                filtered_submodules[module_path] = self.submodule_dict[module_path]
        self.submodule_dict = filtered_submodules

    def run(self, n_tokens: int, tokens: TensorType["batch", "seq"]):
        """
        Run the latent caching process.

        Args:
            n_tokens (int): Total number of tokens to process.
            tokens (TensorType["batch", "seq"]): Input tokens.
        """
        token_batches = self.load_token_batches(n_tokens, tokens)

        total_tokens = 0
        total_batches = len(token_batches)
        tokens_per_batch = token_batches[0].numel()

        with tqdm(total=total_batches, desc="Caching latents") as pbar:
            for batch_number, batch in enumerate(token_batches):
                total_tokens += tokens_per_batch

                with torch.no_grad():
                    buffer = {}
                    with self.model.trace(batch):
                        for module_path, submodule in self.submodule_dict.items():
                            buffer[module_path] = submodule.ae.output.save()
                    for module_path, latents in buffer.items():
                        self.cache.add(latents, batch, batch_number, module_path)

                    del buffer
                torch.cuda.empty_cache()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"Total Tokens": f"{total_tokens:,}"})

        print(f"Total tokens processed: {total_tokens:,}")
        self.cache.save()

    def save(self, save_dir, save_tokens: bool = True):
        """
        Save the cached latents to disk.

        Args:
            save_dir (str): Directory to save the latents.
            save_tokens (bool): Whether to save the dataset tokens used to generate the cache. Defaults to True.
        """
        for module_path in self.cache.latent_locations.keys():
            output_file = f"{save_dir}/{module_path}.safetensors"

            data = {
                "locations": self.cache.latent_locations[module_path],
                "activations": self.cache.latent_activations[module_path],
            }
            if save_tokens:
                data["tokens"] = self.cache.tokens[module_path]

            save_file(data, output_file)

    def _generate_split_indices(self, n_splits):
        """
        Generate indices for splitting the latent space.

        Args:
            n_splits (int): Number of splits to generate.

        Returns:
            List[Tuple[int, int]]: List of start and end indices for each split.
        """
        boundaries = torch.linspace(0, self.width, steps=n_splits + 1).long()

        # Adjust end by one
        return list(zip(boundaries[:-1], boundaries[1:] - 1))

    def save_splits(self, n_splits: int, save_dir, save_tokens: bool = True):
        """
        Save the cached non-zero latent activations and locations in splits.

        Args:
            n_splits (int): Number of splits to generate.
            save_dir (str): Directory to save the splits.
            save_tokens (bool): Whether to save the dataset tokens used to generate the cache. Defaults to True.
        """
        split_indices = self._generate_split_indices(n_splits)
        for module_path in self.cache.latent_locations.keys():
            latent_locations = self.cache.latent_locations[module_path]
            latent_activations = self.cache.latent_activations[module_path]
            tokens = self.cache.tokens[module_path].numpy()

            latent_indices = latent_locations[:, 2]

            for start, end in split_indices:
                mask = (latent_indices >= start) & (latent_indices <= end)

                masked_activations = latent_activations[mask].half().numpy()

                masked_locations = latent_locations[mask].numpy()

                # Optimization to reduce the max value to enable a smaller dtype
                masked_locations[:, 2] = masked_locations[:, 2] - start.item()

                if (
                    masked_locations[:, 2].max() < 2**16
                    and masked_locations[:, 0].max() < 2**16
                ):
                    masked_locations = masked_locations.astype(np.uint16)
                else:
                    masked_locations = masked_locations.astype(np.uint32)

                module_dir = f"{save_dir}/{module_path}"
                os.makedirs(module_dir, exist_ok=True)

                output_file = f"{module_dir}/{start}_{end}.safetensors"

                split_data = {
                    "locations": masked_locations,
                    "activations": masked_activations,
                }
                if save_tokens:
                    split_data["tokens"] = tokens

                save_file(split_data, output_file)

    def save_config(self, save_dir: str, cfg: CacheConfig, model_name: str):
        """
        Save the configuration for the cached latents.

        Args:
            save_dir (str): Directory to save the configuration.
            cfg (CacheConfig): Configuration object.
            model_name (str): Name of the model.
        """
        for module_path in self.cache.latent_locations.keys():
            config_file = f"{save_dir}/{module_path}/config.json"
            with open(config_file, "w") as f:
                config_dict = cfg.to_dict()
                config_dict["model_name"] = model_name
                json.dump(config_dict, f)
