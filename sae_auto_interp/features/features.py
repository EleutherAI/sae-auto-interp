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

from .example import Example


from .utils import display
from ..load.activations import pool_max_activation_slices, get_non_activating_tokens

@dataclass
class Feature:
    module_name: int
    feature_index: int
    
    def __repr__(self) -> str:
        return f"{self.module_name}_feature{self.feature_index}"
    
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
    def from_locations(
        feature: Feature,
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

        record = FeatureRecord(feature)
        
        processed_tokens, processed_activations = pool_max_activation_slices(
            feature_locations, feature_activations, tokens, ctx_len=20, k=max_examples
        )

        record.examples = Example.prepare_examples(processed_tokens, processed_activations)
        
        # sampler(self)

        # # POSTPROCESSING

        # if n_random > 0:
        #     random_tokens = get_non_activating_tokens(
        #         feature_locations, tokens, n_random
        #     )

        #     self.random_examples = self.prepare_examples(
        #         random_tokens, torch.zeros_like(random_tokens),
        #     )

        # # Load processed data if a directory is provided
        # if processed_dir:
        #     self.load_processed(processed_dir)

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



