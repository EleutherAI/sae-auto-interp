import torch
from typing import Tuple, List, Dict
from tqdm import tqdm
import psutil
import orjson
import os
from collections import defaultdict
from ..utils import load_tokenized_data

from .. import cache_config as CONFIG

class Buffer:
    def __init__(
        self
    ):
        self.feature_locations = defaultdict(list)
        self.feature_activations = defaultdict(list)
        self.saved = False

    def add(
        self,
        latents: torch.Tensor,
        batch_number: int,
        layer: int
    ):
        feature_locations, feature_activations = self.get_nonzeros(latents)
        feature_locations = feature_locations.cpu()
        feature_activations = feature_activations.cpu()

        feature_locations[:,0] += batch_number * CONFIG.minibatch_size
        self.feature_locations[layer].append(feature_locations)
        self.feature_activations[layer].append(feature_activations)
        
    def save(self):

        if self.saved:
            return

        for layer in self.feature_locations.keys():
            self.feature_locations[layer] = torch.cat(self.feature_locations[layer], dim=0)
            self.feature_activations[layer] = torch.cat(self.feature_activations[layer], dim=0)
        
        self.saved = True


    def get_nonzeros(self, latents):
        nonzero_feature_locations = torch.nonzero(latents.abs() > 1e-5)
        nonzero_feature_activations = latents[latents.abs() > 1e-5]

        return nonzero_feature_locations, nonzero_feature_activations
    

class FeatureCache:

    def __init__(
        self,
        model, 
        submodule_dict
    ):  
        self.model = model
        self.submodule_dict = submodule_dict
        self.buffer = Buffer()


    def check_memory(self, threshold=0.9):
        # Get memory usage as a percentage
        memory_usage = psutil.virtual_memory().percent / 100.0
        return memory_usage > threshold


    def load_token_batches(self, minibatch_size=20):
        tokens = load_tokenized_data(self.model.tokenizer)

        max_batches = CONFIG.n_tokens // CONFIG.batch_len
        tokens = tokens[:max_batches]
        
        n_mini_batches = len(tokens) // minibatch_size

        token_batches = [
            tokens[minibatch_size * i : minibatch_size * (i + 1), :] 
            for i in range(n_mini_batches)
        ]

        return token_batches
    
    
    def run(self):
        token_batches = self.load_token_batches(CONFIG.minibatch_size)

        total_tokens = 0
        total_batches = len(token_batches)

        with tqdm(total=total_batches, desc="Caching features") as pbar:
            
            for batch_number, batch in enumerate(token_batches):

                if self.check_memory(threshold=0.95):
                    print("Memory usage high. Stopping processing.")
                    break

                batch_tokens = batch.numel()
                total_tokens += batch_tokens

                with torch.no_grad():
                    buffer = {}

                    with self.model.trace(batch, scan=False, validate=False):
                        for layer, submodule in self.submodule_dict.items():
                            buffer[layer] = submodule.ae.output.save()

                    for layer, latents in buffer.items():
                        self.buffer.add(latents, batch_number, layer)

                    del buffer
                    torch.cuda.empty_cache()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({'Total Tokens': f'{total_tokens:,}'})

        print(f"Total tokens processed: {total_tokens:,}")


    def _generate_split_indices(self, n_splits):
        return torch.arange(0, CONFIG.n_features).chunk(n_splits)

    def save_splits(self, n_splits, layer, save_dir):
        self.buffer.save()

        split_indices = self._generate_split_indices(n_splits)
        feature_locations = self.buffer.feature_locations[layer]
        feature_activations = self.buffer.feature_activations[layer]

        # Extract third elements
        third_elements = feature_locations[:, 2]

        for split_index, split in enumerate(split_indices):
            # Create mask for this split
            mask = torch.isin(third_elements, split)
            
            # Mask and save feature locations
            masked_locations = feature_locations[mask]
            location_output_file = os.path.join(save_dir, f"layer{layer}_split_{split_index}_locations.pt")
            torch.save(masked_locations, location_output_file)

            # Mask and save feature activations
            masked_activations = feature_activations[mask]
            activation_output_file = os.path.join(save_dir, f"layer{layer}_split_{split_index}_activations.pt")
            torch.save(masked_activations, activation_output_file)

    def save_selected_features(
        self, 
        feature_list, 
        layer, 
        save_dir
    ):
        self.buffer.save()

        feature_locations = self.buffer.feature_locations[layer]
        feature_activations = self.buffer.feature_activations[layer]

        # Extract third elements
        third_elements = feature_locations[:, 2]

        # Create mask for this split
        mask = torch.isin(third_elements, feature_list)
        
        # Mask and save feature locations
        masked_locations = feature_locations[mask]
        location_output_file = os.path.join(save_dir, f"layer{layer}_locations.pt")
        torch.save(masked_locations, location_output_file)

        # Mask and save feature activations
        masked_activations = feature_activations[mask]
        activation_output_file = os.path.join(save_dir, f"layer{layer}_activations.pt")
        torch.save(masked_activations, activation_output_file)