from dataclasses import dataclass
from typing import List, Tuple
import torch
from tqdm import tqdm
import orjson
import blobfile as bf
from collections import defaultdict
from typing import List, Callable
from ..logger import logger
from torch import Tensor
from .sampling import default_sampler

from .utils import display
from .activations import pool_max_activation_slices, get_non_activating_tokens

@dataclass
class Feature:
    module_name: int
    feature_index: int
    
    def __repr__(self) -> str:
        return f"{self.module_name}_feature{self.feature_index}"
    
@dataclass
class Example:
    tokens: List[int]
    activations: List[float]
    
    def __hash__(self) -> int:
        return hash(tuple(self.tokens))

    def __eq__(self, other) -> bool:
        return self.tokens == other.tokens
    
    def decode(self, tokenizer):
        self.str_toks = tokenizer.batch_decode(self.tokens)
        return self.str_toks

    @property
    def max_activation(self):
        return max(self.activations)
    
    @property
    def text(self):
        return "".join(self.str_toks)

    
class FeatureRecord:

    def __init__(
        self,
        feature: Feature,
    ):
        self.feature = feature

    @property
    def max_activation(self):
        return self.examples[0].max_activation
    
    def prepare_examples(self, tokens, activations):
        return [
            Example(
                tokens=toks,
                activations=acts,
            )
            for toks, acts in zip(
                tokens, 
                activations
            )
        ]

    @classmethod
    def from_tensor(
        cls,
        tokens: Tensor, 
        module_name: str,
        raw_dir: str,
        selected_features: List[int] = None,
        **kwargs
    ) -> List["FeatureRecord"]:
        """
        Loads a list of records from a tensor of locations and activations. Pass
        in a proccessed_dir to load a feature's processed data.
        """
        
        # Build location paths
        locations_path = f"{raw_dir}/{module_name}_locations.pt"
        activations_path = f"{raw_dir}/{module_name}_activations.pt"
        
        # Load tensor
        locations = torch.load(locations_path)
        activations = torch.load(activations_path)

        # Get unique features to load into records
        features = torch.unique(locations[:, 2])

        # Filter selected features
        if selected_features is not None:
            selected_features_tensor = torch.tensor(selected_features)
            features = features[torch.isin(features, selected_features_tensor)]
        
        records = []

        for feature_index in tqdm(features, desc=f"Loading features from tensor for layer {module_name}"):
            
            record = cls(
                Feature(
                    module_name=module_name, 
                    feature_index=feature_index.item()
                )
            )
            
            mask = locations[:, 2] == feature_index
            # Discard the feature dim
            feature_locations = locations[mask][:,:2]
            feature_activations = activations[mask]
        
            try:
                record.from_locations(
                    tokens,
                    feature_locations,
                    feature_activations,
                    **kwargs
                )
            except ValueError as e:
                logger.error(f"Error loading feature {record.feature}: {e}")
                continue

            records.append(record)

        return records
    
    
    def from_locations(
        self,
        tokens: Tensor, 
        feature_locations: Tensor,
        feature_activations: Tensor,
        min_examples: int = 200,
        max_examples: int = 2_000,
        sampler: Callable = default_sampler,
        processed_dir: str = None, 
        n_random: int = 0,
    ):
        """
        Loads a single record from a tensor of locations and activations.
        """
        
        processed_tokens, processed_activations = pool_max_activation_slices(
            feature_locations, feature_activations, tokens, ctx_len=20, k=max_examples
        )

        if len(processed_tokens) < min_examples:
            logger.error(f"Feature {self.feature} has fewer than {min_examples} examples.")
            raise ValueError(f"Feature {self.feature} has fewer than {min_examples} examples.")

        self.examples = self.prepare_examples(processed_tokens, processed_activations)
        
        sampler(self)

        # POSTPROCESSING

        if n_random > 0:
            random_tokens = get_non_activating_tokens(
                feature_locations, tokens, n_random
            )

            self.random_examples = self.prepare_examples(
                random_tokens, torch.zeros_like(random_tokens),
            )

        # Load processed data if a directory is provided
        if processed_dir:
            self.load_processed(processed_dir)

    def display(self, tokenizer, n=10):
        for example in self.examples[:n]:
            example.decode(tokenizer)
        display(self.examples[:n])

    def load_processed(self, directory: str):
        path = f"{directory}/{self.feature}.json"

        with bf.BlobFile(path, "rb") as f:
            processed_data = orjson.loads(f.read())
            self.__dict__.update(processed_data)
    
    def save(self, directory: str, save_examples=False):
        path = f"{directory}/{self.feature}.json"
        serializable = self.__dict__

        if not save_examples:
            serializable.pop("examples")
            serializable.pop("train")
            serializable.pop("test")

        serializable.pop("feature")
        with bf.BlobFile(path, "wb") as f:
            f.write(orjson.dumps(serializable))

