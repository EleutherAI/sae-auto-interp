from typing import Dict, List
from ..features import FeatureRecord
from abc import ABC, abstractmethod


import torch
import torch.nn.functional as F
from collections import defaultdict

import umap 
from sklearn.neighbors import NearestNeighbors

import numpy as np
from typing import List
from ..features import Example, FeatureRecord


class Transform(ABC):

    @abstractmethod
    def refresh(self, **kwargs):
        """
        Refresh the stats object state. Used for updating the current 
        logit cache, activation depth, etc.
        """

        pass
    
    @abstractmethod
    def compute(self, records: List[FeatureRecord], *args, **kwargs):
        """
        Compute the stats for all records and store them in the record.
        """

        pass

class CombinedTransform(Transform):

    def __init__(self, **kwargs):
        self._objs: Dict[str, Transform] = kwargs

    def refresh(self, **kwargs):
        for obj in self._objs.values():
            obj.refresh(**kwargs)

    def compute(self, records: List[FeatureRecord], *args, **kwargs):
        for obj in self._objs.values():
            obj.compute(records, *args, **kwargs)


class Logits(Transform):
    collated = True

    def __init__(self, 
        tokenizer, 
        k=10,
        W_U = None,
    ):
        self.tokenizer = tokenizer
        self.k = k
        self.W_U = W_U

    def refresh(self, W_dec=None, **kwargs):
        self.W_dec = W_dec
        
    def compute(self, records, **kwargs):

        feature_indices = [
            record.feature.feature_index 
            for record in records
        ]
        
        narrowed_logits = torch.matmul(
            self.W_U, 
            self.W_dec[:,feature_indices]
        )

        top_logits = torch.topk(
            narrowed_logits, self.k, dim=0
        ).indices

        per_example_top_logits = top_logits.T

        for record_index, record in enumerate(records):

            record.top_logits = \
                self.tokenizer.batch_decode(
                    per_example_top_logits[record_index]
                )
            


def nonzero(example: Example) -> int:
    return np.count_nonzero(example.activations)

def top_tok(example: Example) -> str:
    return example.tokens[np.argmax(example.activations)].item()

class Unigram(Transform):
    def __init__(
            self, 
            k: int = 10
        ):
        self.k = k

    def refresh(self, k: int = None, **kwargs):
        if k is not None:
            self.k = k
            
    def compute(self, records: List[FeatureRecord], **kwargs):
        for record in records:
            self._compute(record)
    
    def _compute(self, record: FeatureRecord):
        avg_nonzero = []
        top_tokens = []

        for example in record.examples[:self.k]:
            avg_nonzero.append(nonzero(example))
            top_tokens.append(top_tok(example))

        record.n_unique = len(set(top_tokens))
        
        avg_nonzero = np.mean(avg_nonzero)
        record.avg_nonzero = float(avg_nonzero)


def cos(matrix, selected_features=[0]):
    
    a = matrix[:,selected_features]
    b = matrix   

    a = F.normalize(a, p=2, dim=0)
    b = F.normalize(b, p=2, dim=0)

    cos_sim = torch.mm(a.t(), b)

    return cos_sim

def get_neighbors(submodule_dict, feature_filter, k=10):
    """
    Get the required features for neighbor scoring.

    Returns:
        neighbors_dict: Nested dictionary of modules -> neighbors -> indices, values
        per_layer_features (dict): A dictionary of features per layer
    """

    neighbors_dict = defaultdict(dict)
    per_layer_features = {}
    
    for module_path, submodule in submodule_dict.items():
        selected_features = feature_filter.get(module_path, False)
        if not selected_features:
            continue

        W_D = submodule.ae.autoencoder._module.decoder.weight
        cos_sim = cos(W_D, selected_features=selected_features)
        top = torch.topk(cos_sim, k=k)

        top_indices = top.indices   
        top_values = top.values

        for i, (indices, values) in enumerate(zip(top_indices, top_values)):
            neighbors_dict[module_path][i] = {
                "indices": indices.tolist()[1:],
                "values": values.tolist()[1:]
            }
        
        per_layer_features[module_path] = torch.unique(top_indices).tolist()

    return neighbors_dict, per_layer_features


class UmapNeighbors(Transform):

    def __init__(
        self, 
        n_neighbors: int = 15, 
        metric: str = 'cosine', 
        min_dist: float = 0.05, 
        n_components: int = 2, 
        random_state: int = 42,
        **kwargs
    ):
        self.umap_model = umap.UMAP(
            n_neighbors=n_neighbors, 
            metric=metric, 
            min_dist=min_dist, 
            n_components=n_components, 
            random_state=random_state,
            **kwargs
        )

    def refresh(self, W_dec=None, **kwargs):
        self.embedding = \
            self.umap_model.fit_transform(W_dec)

    def compute(self, records, *args, **kwargs): 
        for record in records:
            self._compute(record, *args, **kwargs)

    def _compute(self, record, *args, **kwargs):
        # Increment n_neighbors to account for query
        n_neighbors = n_neighbors + 1
        feature_index = record.feature.feature_index
        query = self.embedding[feature_index]

        nn_model = NearestNeighbors(n_neighbors=n_neighbors)
        nn_model.fit(self.embedding)

        distances, indices = nn_model.kneighbors([query])

        neighbors = {
            'distances': distances[0,1:].tolist(),
            'indices': indices[0,1:].tolist()
        }

        record.neighbors = neighbors

class CosNeigbors(Transform):
    pass