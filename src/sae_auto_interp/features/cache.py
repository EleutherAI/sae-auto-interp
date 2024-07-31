import torch
from typing import Dict
from tqdm import tqdm
from torchtyping import TensorType
import os
from collections import defaultdict
from safetensors.torch import save_file

from ..config import CacheConfig

class Cache:
    """
    The Buffer class stores feature locations and activations for modules.
    """

    def __init__(
        self,
        filters: Dict[str, TensorType["indices"]] = None,
        batch_size: int = 64
    ):
        self.feature_locations = defaultdict(list)
        self.feature_activations = defaultdict(list)
        self.filters = filters
        self.batch_size = batch_size

    def add(
        self,
        latents: TensorType["batch", "sequence", "feature"],
        batch_number: int,
        module_path: str
    ):
        """
        Add the latents from a module to the buffer
        """
        feature_locations, feature_activations = \
            self.get_nonzeros(latents, module_path)
        feature_locations = feature_locations.cpu()
        feature_activations = feature_activations.cpu()

        feature_locations[:,0] += batch_number * self.batch_size
        self.feature_locations[module_path].append(feature_locations)
        self.feature_activations[module_path].append(feature_activations)
        
    def save(self):
        """
        Concatenate the feature locations and activations
        """

        for module_path in self.feature_locations.keys():
            self.feature_locations[module_path] = \
                torch.cat(self.feature_locations[module_path], dim=0)
            
            self.feature_activations[module_path] = \
                torch.cat(self.feature_activations[module_path], dim=0)

    def get_nonzeros(
        self, 
        latents: TensorType["batch", "seq", "feature"], 
        module_path: str
    ):
        """
        Get the nonzero feature locations and activations
        """

        nonzero_feature_locations = torch.nonzero(latents.abs() > 1e-5)
        nonzero_feature_activations = latents[latents.abs() > 1e-5]

        # Return all nonzero features if no filter is provided
        if self.filters is None:
            return nonzero_feature_locations, nonzero_feature_activations
        
        # Return only the selected features if a filter is provided
        else:
            selected_features = self.filters[module_path]
            mask = torch.isin(
                nonzero_feature_locations[:, 2], 
                selected_features
            )

            return nonzero_feature_locations[mask], \
                nonzero_feature_activations[mask]


class FeatureCache:

    def __init__(
        self,
        model, 
        submodule_dict: Dict,
        batch_size: int,
        filters: Dict[str, TensorType["indices"]] = None,
    ):  
        self.model = model
        self.submodule_dict = submodule_dict

        self.batch_size = batch_size
        self.width = list(submodule_dict.values())[0].ae.width

        self.cache = Cache(filters, batch_size=batch_size)
        if filters is not None:
            self.filter_submodules(filters)

        print(submodule_dict.keys())

    def load_token_batches(self, n_tokens: int, tokens: TensorType["batch", "sequence"]):
        
        max_batches = n_tokens // tokens.shape[1]
        tokens = tokens[:max_batches]
        
        n_mini_batches = len(tokens) // self.batch_size

        token_batches = [
            tokens[self.batch_size * i : self.batch_size * (i + 1), :] 
            for i in range(n_mini_batches)
        ]

        return token_batches
    
    def filter_submodules(self, filters: Dict[str, TensorType["indices"]]):
        filtered_submodules = {}
        for module_path in self.submodule_dict.keys():
            if module_path in filters:
                filtered_submodules[module_path] = self.submodule_dict[module_path]
        self.submodule_dict = filtered_submodules

    def run(self, n_tokens: int, tokens: TensorType["batch", "seq"]):
        token_batches = self.load_token_batches(n_tokens, tokens)

        total_tokens = 0
        total_batches = len(token_batches)
        tokens_per_batch = token_batches[0].numel()

        with tqdm(total=total_batches, desc="Caching features") as pbar:
            
            for batch_number, batch in enumerate(token_batches):
                total_tokens += tokens_per_batch

                with torch.no_grad():
                    buffer = {}

                    with self.model.trace(batch, scan=False, validate=False):
                        for module_path, submodule in self.submodule_dict.items():
                            buffer[module_path] = submodule.ae.output.save()

                    for module_path, latents in buffer.items():
                        self.cache.add(
                            latents, 
                            batch_number, 
                            module_path
                        )

                    del buffer
                    torch.cuda.empty_cache()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({'Total Tokens': f'{total_tokens:,}'})

        print(f"Total tokens processed: {total_tokens:,}") 
        self.cache.save()

    def save(self, save_dir):

        for module_path in self.cache.feature_locations.keys():

            output_file = f"{save_dir}/{module_path}.safetensors"

            data = {
                "locations" : self.cache.feature_locations[module_path],
                "activations" : self.cache.feature_activations[module_path]
            }

            save_file(data, output_file)

    def _generate_split_indices(self, n_splits):
        boundaries = torch.linspace(0, self.width, steps=n_splits+1).long()

        # Adjust end by one
        return list(zip(boundaries[:-1], boundaries[1:] - 1))

    def save_splits(self, n_splits: int, save_dir):

        split_indices = self._generate_split_indices(n_splits)

        for module_path in self.cache.feature_locations.keys():

            feature_locations = self.cache.feature_locations[module_path]
            feature_activations = self.cache.feature_activations[module_path]

            features = feature_locations[:, 2]

            for start, end in split_indices:
                
                mask = (features >= start) & (features < end)
                
                masked_locations = feature_locations[mask]
                masked_activations = feature_activations[mask]

                module_dir = f"{save_dir}/{module_path}"
                os.makedirs(module_dir, exist_ok=True)

                output_file = f"{module_dir}/{start}_{end}.safetensors"

                split_data = {
                    "locations" : masked_locations,
                    "activations" : masked_activations
                }

                save_file(split_data, output_file)