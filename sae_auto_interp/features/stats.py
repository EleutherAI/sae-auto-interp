from typing import List
from collections import defaultdict
import umap
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import skew, kurtosis
import spacy
from tqdm import tqdm
import torch
from typing import Dict
from .backends import UmapBackend, LogitBackend

class Stat:
    _backends = {}

    @classmethod
    def get_backend(cls, backend_name: str):
        return cls._backends.get(backend_name)

    @classmethod
    def set_backend(cls, backend_name: str, backend):
        cls._backends[backend_name] = backend

    def compute(self, record, *args, **kwargs):
        pass


class CombinedStat(Stat):
    def __init__(self, **kwargs):
        self._objs: Dict[str, Stat] = kwargs

    def refresh(self, **kwargs):
        for _, backend in self._backends.items():
            backend.refresh(**kwargs)

    def compute(self, records, *args, **kwargs):
        for record in tqdm(records):
            for obj in self._objs.values():
                obj.compute(record, *args, **kwargs)


class Neighbors(Stat):
    backend = UmapBackend

    def __init__(self):
        # Sets class backend
        self.set_backend("umap", self.backend())

    def refresh(self, W_dec=None):
        backend = self.get_backend("umap")
        backend.refresh(W_dec)

    def compute(self, record, *args, **kwargs):
        # Increment n_neighbors to account for query
        n_neighbors = n_neighbors + 1
        feature_index = record.feature.feature_index
        query = self.embedding[feature_index]
        nn_model = NearestNeighbors(n_neighbors=n_neighbors)
        embedding = self.get_backend("umap").embedding
        nn_model.fit(embedding)

        distances, indices = nn_model.kneighbors([query])

        neighbors = {
            'distances': distances[0,1:].tolist(),
            'indices': indices[0,1:].tolist()
        }

        record.neighbors = neighbors


class Skew(Stat):
    backend = LogitBackend

    def __init__(self):
        self.set_backend("logit", self.backend())

    def refresh(self, W_U=None, W_dec=None):
        backend = self.get_backend("logit")
        backend.refresh(W_U, W_dec)

    def compute(self, record, *args, **kwargs):
        backend = self.get_backend("logit")
        logits = backend.logits[record.feature.feature_index, :]
        record.skewness = float(skew(logits))


class Kurtosis(Stat):
    backend = LogitBackend

    def __init__(self):
        self.set_backend("logit", self.backend())

    def refresh(self, W_U=None, W_dec=None):
        backend = self.get_backend("logit")
        backend.refresh(W_U, W_dec)

    def compute(self, record, *args, **kwargs):
        backend = self.get_backend("logit")
        logits = backend.logits[record.feature.feature_index, :]
        record.kurtosis = float(kurtosis(logits))
    
class TopLogits(Stat):
    backend = LogitBackend

    def __init__(self, k=10):
        self.set_backend("logit", self.backend())
        self.k = k

    def refresh(self, W_U=None, W_dec=None):
        backend = self.get_backend("logit")
        backend.refresh(W_U, W_dec)

    def compute(self, record, *args, **kwargs):
        backend = self.get_backend("logit")
        logits = backend.logits[record.feature.feature_index, :]
        top_logits = torch.topk(logits, self.k)
        record.top_logits = [
            self.model.tokenizer.decode([token]) 
            for token in top_logits.indices
        ]


class Activations(Stat):
    def __init__(self, lemmatize=False, top_activating_k=100):
        self.lemmatize = lemmatize
        self.top_activating_k = top_activating_k
        
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

    def get_top_activating_tokens(
        self,
        examples,
        k
    ):
        """
        Get the top token for the top k examples
        """
        top_k = examples[:k]
        top_indices = [
            np.argmax(example.activations)
            for example in top_k
        ]
        top_tokens = [
            example.str_toks[index]
            for example, index 
            in zip(top_k, top_indices)
        ]

        return top_tokens
    
    def n_above_zero(self, examples):
        """
        Get number of positive activations for each example
        """
        return [
            int(sum(np.array(example.activations) > 0))
            for example in examples
        ]
    
    def _lemmatize(self, words):

        unique_tokens = list(set(words))
        lowercase_tokens = [
            token.lower().strip() 
            for token in unique_tokens
        ]
        alpha_tokens = [
            token for token 
            in lowercase_tokens 
            if token.isalpha()
        ]
        text_for_spacy = " ".join(alpha_tokens)

        doc = self.nlp(text_for_spacy)

        lemmatized_tokens = [token.lemma_ for token in doc]
        return lemmatized_tokens

    def compute(self, record):
        top_tokens = self.get_top_activating_tokens(
            record.examples, self.top_activating_k
        )
        n_above_zero = self.n_above_zero(record.examples)
        
        record.mean_n_activations = float(np.average(n_above_zero))

        if self.lemmatize:
            record.lemmas = self._lemmatize(top_tokens)
