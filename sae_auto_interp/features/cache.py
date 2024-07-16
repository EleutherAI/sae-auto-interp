import torch
from typing import Dict
from tqdm import tqdm
import psutil
from torchtyping import TensorType
import os
from collections import defaultdict

class Buffer:
    """
    The Buffer class stores feature locations and activations for modules.
    """

    def __init__(
        self,
        filters: Dict[str, TensorType["indices"]] = None,
        minibatch_size: int = 64
    ):
        self.feature_locations = defaultdict(list)
        self.feature_activations = defaultdict(list)
        self.saved = False
        self.filters = filters
        self.minibatch_size = minibatch_size

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

        feature_locations[:,0] += batch_number * self.minibatch_size
        self.feature_locations[module_path].append(feature_locations)
        self.feature_activations[module_path].append(feature_activations)
        
    def save(self):
        """
        Concatenate the feature locations and activations
        """

        if self.saved:
            return

        for module_path in self.feature_locations.keys():
            self.feature_locations[module_path] = \
                torch.cat(self.feature_locations[module_path], dim=0)
            
            self.feature_activations[module_path] = \
                torch.cat(self.feature_activations[module_path], dim=0)
        
        self.saved = True

    def get_nonzeros(self, latents, module_path):
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
        minibatch_size: int = 64,
        filters: Dict[str, TensorType["indices"]] = None
    ):  
        self.model = model
        self.submodule_dict = submodule_dict
        # Get the weidth from the first submodule
        first_sae = list(submodule_dict.values())[0].ae

        self.width = first_sae.n_features
        print(f"Feature width: {self.width}")
        self.minibatch_size = minibatch_size
        
        if filters is not None:
            self.filter_submodules(filters)
            self.buffer = Buffer(filters, minibatch_size=minibatch_size)

        else:
            self.buffer = Buffer(minibatch_size=minibatch_size)


    def check_memory(self, threshold=0.9):
        """
        Check memory usage to kick out of training before crash.
        """
        memory_usage = psutil.virtual_memory().percent / 100.0
        return memory_usage > threshold


    def load_token_batches(self, tokens, minibatch_size=20):
        
        max_batches = self.n_tokens // self.seq_len
        tokens = tokens[:max_batches]
        
        n_mini_batches = len(tokens) // minibatch_size

        token_batches = [
            tokens[minibatch_size * i : minibatch_size * (i + 1), :] 
            for i in range(n_mini_batches)
        ]

        return token_batches
    
    def filter_submodules(self, filters):
        filtered_submodules = {}
        for module_path in self.submodule_dict.keys():
            if module_path in filters:
                filtered_submodules[module_path] = self.submodule_dict[module_path]
        self.submodule_dict = filtered_submodules

    def run(self, tokens, n_tokens=10_000_000):
        self.n_tokens = n_tokens
        self.seq_len = tokens.shape[1]
        token_batches = self.load_token_batches(tokens,self.minibatch_size)

        total_tokens = 0
        total_batches = len(token_batches)
        tokens_per_batch = self.minibatch_size * self.seq_len

        with tqdm(total=total_batches, desc="Caching features") as pbar:
            
            for batch_number, batch in enumerate(token_batches):
                total_tokens += tokens_per_batch

                if self.check_memory(threshold=0.95):
                    print("Memory usage critical.")
                    raise MemoryError("Memory usage is critical. Exiting.")

                with torch.no_grad():
                    _buffer = {}

                    with self.model.trace(batch, scan=False, validate=False):
                        for module_path, submodule in self.submodule_dict.items():
                            _buffer[module_path] = submodule.ae.output.save()

                    for module_path, latents in _buffer.items():
                        self.buffer.add(
                            latents, 
                            batch_number, module_path
                        )

                    del _buffer
                    torch.cuda.empty_cache()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({'Total Tokens': f'{total_tokens:,}'})

        print(f"Total tokens processed: {total_tokens:,}") 
        self.buffer.save()

    def _generate_split_indices(self, width, n_splits):
        return torch.arange(0, width).chunk(n_splits)
    
    def save(self, save_dir):

        for module_path in self.buffer.feature_locations.keys():
            location_output_file = os.path.join(save_dir, f"{module_path}_locations.pt")
            activation_output_file = os.path.join(save_dir, f"{module_path}_activations.pt")

            torch.save(self.buffer.feature_locations[module_path], location_output_file)
            torch.save(self.buffer.feature_activations[module_path], activation_output_file)

    def save_splits(self, n_splits, module_path, save_dir):

        split_indices = self._generate_split_indices(n_splits)
        feature_locations = self.buffer.feature_locations[module_path]
        feature_activations = self.buffer.feature_activations[module_path]

        # Extract third elements
        third_elements = feature_locations[:, 2]

        for split_index, split in enumerate(split_indices):
            # Create mask for this split
            mask = torch.isin(third_elements, split)
            
            # Mask and save feature locations
            masked_locations = feature_locations[mask]
            location_output_file = os.path.join(save_dir, f"{module_path}_split_{split_index}_locations.pt")
            torch.save(masked_locations, location_output_file)

            # Mask and save feature activations
            masked_activations = feature_activations[mask]
            activation_output_file = os.path.join(save_dir, f"{module_path}_split_{split_index}_activations.pt")
            torch.save(masked_activations, activation_output_file)

    def save_selected_features(
        self, 
        feature_list, 
        module_path, 
        save_dir
    ):

        feature_locations = self.buffer.feature_locations[module_path]
        feature_activations = self.buffer.feature_activations[module_path]

        # Extract third elements
        third_elements = feature_locations[:, 2]

        # Create mask for this split
        mask = torch.isin(third_elements, feature_list)
        
        # Mask and save feature locations
        masked_locations = feature_locations[mask]
        masked_activations = feature_activations[mask]

        location_output_file = os.path.join(save_dir, f"{module_path}_locations.pt")
        activation_output_file = os.path.join(save_dir, f"{module_path}_activations.pt")

        torch.save(masked_locations, location_output_file)
        torch.save(masked_activations, activation_output_file)