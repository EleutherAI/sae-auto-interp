from typing import List
from collections import defaultdict
import umap
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import skew, kurtosis
import spacy
from tqdm import tqdm
import torch
from typing import Dict, Any, ClassVar


def load_umap(W_dec):
    def _umap_backend():
        umap_model = umap.UMAP(
            n_neighbors=15, 
            metric='cosine', 
            min_dist=0.05, 
            n_components=2, 
            random_state=42
        )

        
        
    return _umap_backend


def load_logits():
    def _logits_backend(W_U=None, W_dec=None):
        logits = torch.matmul(W_U, W_dec).detach().cpu()
        return logits
    return _logits_backend

def load_backend(backend_name: str):
    if backend_name == "umap":
        return load_umap
    elif backend_name == "logits":
        return load_logits
    else:
        return None


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
        self._load_backends()

    def _load_backends(self):
        names = [obj.backend_name for obj in self._objs.values()]
        self._backend_names = set(names)

        for backend_name in self._backend_names:
            

    def refresh(self, **kwargs):
        for backend_name in self._backend_names:
            backend = self.get_backend(backend_name)
            if backend and callable(backend):
                backend(**kwargs)

    def compute(self, records, *args, **kwargs):
        for record in tqdm(records):
            for obj in self._objs.values():
                obj.compute(record, *args, **kwargs)


class Neighbors(Stat):
    def __init__(self, W_dec):
        self.backend_name = "umap"
        self.W_dec = W_dec

    def compute(self, record, *args, **kwargs):
        backend = self.get_backend(self.backend_name)
        if backend:
            # Use the backend to compute
            pass


class Neighbors(Stat):
    backend_name = "umap"

    def __init__(self, W_dec):
        self.W_dec = W_dec
        self.backend = None

    def refresh(self, **kwargs):
        self.backend = self.get_backend(self.backend_name, W_dec=self.W_dec)

    def compute(self, records, n_neighbors=10):
        embedding = self.backend()

        for record in records:
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

        
    
class Logits(Stat):

    def __init__(
        self, 
        model,
        get_skewness=False,
        get_kurtosis=False,
        top_k_logits=None
    ):
        self.model = model
        self.get_skewness = get_skewness
        self.get_kurtosis = get_kurtosis
        self.top_k_logits = top_k_logits

    def refresh(self, W_dec=None):
        self.load_top_logits(W_dec)

    def load_top_logits(self, W_dec):
        W_U = self.model.transformer.ln_f.weight * self.model.lm_head.weight
        self.logits = torch.matmul(W_dec.T, W_U.T).detach().cpu()
        
    def compute(self, record):
        feature_index = record.feature.feature_index
        logits = self.logits[feature_index, :]

        if self.get_skewness:
            record.skewness = float( skew(logits))

        if self.get_kurtosis:
            record.kurtosis = float(kurtosis(logits))

        if self.top_k_logits is not None:
            top_logits = torch.topk(logits, self.top_k_logits)
            record.top_logits = [
                self.model.tokenizer.decode([token]) 
                for token in top_logits.indices
            ]



class Activations(Stat):
    def __init__(self, get_lemmas=False, top_activating_k=100):
        self.get_lemmas = get_lemmas
        self.top_activating_k = top_activating_k
        self.nlp = spacy.load("en_core_web_sm")

    def get_top_activating_tokens(
        self,
        record,
        k
    ):
        examples = record.examples

        # Examples are sorted, so get top k
        top_k = examples[:k]

        # Get highest activation
        top_indices = [
            np.argmax(example.activations)
            for example in top_k
        ]

        # Get the respective highest token
        top_tokens = [
            example.str_toks[index]
            for example, index in zip(top_k, top_indices)
        ]

        return top_tokens
    
    def n_above_zero(self, examples):
        return [
            int(sum(np.array(example.activations) > 0))
            for example in examples
        ]
    
    def lemmatize(self, words):

        # Step 1: Remove duplicates
        unique_tokens = list(set(words))

        # Step 2: Convert to lowercase
        lowercase_tokens = [token.lower().strip() for token in unique_tokens]

        # Step 3: Remove non-alphabetic tokens
        alpha_tokens = [token for token in lowercase_tokens if token.isalpha()]

        # Step 4: Join into a single string
        text_for_spacy = " ".join(alpha_tokens)

        doc = self.nlp(text_for_spacy)

        lemmatized_tokens = [token.lemma_ for token in doc]
        return lemmatized_tokens

    def compute(self, record):
        top_tokens = self.get_top_activating_tokens(record, self.top_activating_k)
        n_above_zero = self.n_above_zero(record.examples)
        
        record.mean_n_activations = float(np.average(n_above_zero))

        if self.get_lemmas:
            record.lemmas = self.lemmatize(top_tokens)
