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
import random

from .. import cache_config as CONFIG

@dataclass
class Feature:
    layer_index: int
    feature_index: int
    
    def __repr__(self) -> str:
        return f"layer{self.layer_index}_feature{self.feature_index}"
    
@dataclass
class Example:
    tokens: List[int]
    activations: List[float]
    str_toks: List[str] = None

    def __hash__(self) -> int:
        if self.str_toks is None:
            raise ValueError("Cannot hash examples without decoding.")
        
        return hash(self.text)

    def __eq__(self, other) -> bool:
        if self.str_toks is None:
            raise ValueError("Cannot compare examples without decoding.")
        
        return self.text == other.text

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
    
    @staticmethod
    def display(
        examples=None
    ) -> str:
        assert hasattr(examples[0], "str_toks"), \
            "Examples must have be detokenized to display."

        from IPython.core.display import display, HTML

        def _to_string(tokens, activations):
            result = []
            i = 0
            while i < len(tokens):
                if activations[i] > 0:
                    result.append("<mark>")
                    while i < len(tokens) and activations[i] > 0:
                        result.append(tokens[i])
                        i += 1
                    result.append("</mark>")
                else:
                    result.append(tokens[i])
                    i += 1
            return "".join(result)
        
        strings = [
            _to_string(
                example.str_toks, 
                example.activations
            ) 
            for example in examples
        ]

        display(HTML("<br><br>".join(strings)))
    
    @classmethod
    def from_tensor(
        cls,
        tokens: Tensor, 
        layer_index: int,
        raw_dir: str,
        selected_features: List[int] = None,
        **kwargs
    ) -> List["FeatureRecord"]:
        """
        Loads a list of records from a tensor of locations and activations. Pass
        in a proccessed_dir to load a feature's processed data.

        Args:
            tokens: Tokenized data from caching
            tokenizer: Tokenizer from the model
            layer: Layer index
            locations_path: Path to the locations tensor
            activations_path: Path to the activations tensor
            processed_dir: Path to the processed data
            max_examples: Maximum number of examples to load per record

        Returns:
            List of FeatureRecords
        """
        
        # Build location paths
        locations_path = f"{raw_dir}/layer{layer_index}_locations.pt"
        activations_path = f"{raw_dir}/layer{layer_index}_activations.pt"
        
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


        for feature_index in tqdm(features, desc=f"Loading features from tensor for layer {layer_index}"):
            
            record = cls(
                Feature(
                    layer_index=layer_index, 
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
        processed_dir: str = None, 
        tokenizer: Callable = None,
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

        self.examples = self._prepare(processed_tokens, processed_activations)

        # POSTPROCESSING

        if n_random > 0:
            random_tokens = get_non_activating_tokens(
                feature_locations, tokens, n_random
            )

            self.random_examples = self._prepare(
                random_tokens, [-1] * n_random, 
                tokenizer=tokenizer
            )

        # Load processed data if a directory is provided
        if processed_dir:
            self.load_processed(processed_dir)



    def _prepare(self, tokens, activations, tokenizer=None):
        # Messy but I feel like millions of conditionals is slower.
        if tokenizer is None:
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
        else:
            return [
                Example(
                    tokens=toks,
                    activations=acts,
                    str_toks=tokenizer.batch_decode(toks)
                )
                for toks, acts in zip(
                    tokens, 
                    activations
                )
            ]

    def load_processed(self, directory: str):
        path = f"{directory}/{self.feature}.json"
        with bf.BlobFile(path, "rb") as f:
            processed_data = orjson.loads(f.read())
            feature_dict = processed_data.pop("feature")
            self.feature = Feature(**feature_dict)
            self.__dict__.update(processed_data)
    
    def save(self, directory: str, save_examples=False):
        path = f"{directory}/{self.feature}.json"
        serializable = self.__dict__

        if not save_examples:
            serializable.pop("examples")

        with bf.BlobFile(path, "wb") as f:
            f.write(orjson.dumps(serializable))


def pool_max_activation_slices(
    locations, activations, tokens, ctx_len, k=10
):
    batch_len, seq_len = tokens.shape

    sparse_activations = torch.sparse_coo_tensor(
        locations.t(), activations, (batch_len, seq_len)
    )
    dense_activations = sparse_activations.to_dense()

    unique_batch_pos = torch.unique(locations[:,0])
    token_batches = tokens[unique_batch_pos]
    dense_activations = dense_activations[unique_batch_pos]

    avg_pools = torch.nn.functional.max_pool1d(
        dense_activations, kernel_size=ctx_len, stride=ctx_len
    )

    non_zero = avg_pools != 0
    avg_pools = avg_pools[non_zero]

    activation_windows = dense_activations.unfold(1, ctx_len, ctx_len).reshape(-1, ctx_len)
    token_windows = token_batches.unfold(1, ctx_len, ctx_len).reshape(-1, ctx_len)

    k = min(k, len(avg_pools))
    top_indices = torch.topk(avg_pools.flatten(), k).indices

    activation_windows = activation_windows[top_indices]
    token_windows = token_windows[top_indices]

    return token_windows, activation_windows


def get_non_activating_tokens(
    locations, tokens, n_to_find, ctx_len=20
):
    unique_batch_pos = torch.unique(locations[:,0])
    taken = set(unique_batch_pos.tolist())
    free = []
    value = 0
    while value < n_to_find:
        if value not in taken:
            free.append(value)
            value += 1
    return tokens[free, 10:10+ctx_len]