from typing import List
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import torch
from typing import Dict
import torch.nn.functional as F

import umap
import random


class Stat:

    def refresh(self, **kwargs):
        pass

    def compute(self, record, *args, **kwargs):
        pass

class CombinedStat(Stat):
    def __init__(self, **kwargs):
        self._objs: Dict[str, Stat] = kwargs

    def refresh(self, **kwargs):
        for obj in self._objs.values():
            obj.refresh(**kwargs)

    def compute(self, records, save_dir=None, *args, **kwargs):

        records = [
            record 
            for record in records 
            if type(record) is not str
        ]

        for obj in self._objs.values():
            if obj.collated:
                obj.compute(records, *args, **kwargs)

        for record in records:
            for obj in self._objs.values():
                if not obj.collated:
                    obj.compute(record, *args, **kwargs)
                    
                    if save_dir is not None:
                        record.save(save_dir)


class Neighbors(Stat):
    collated = False

    def refresh(self, W_dec=None, **kwargs):
        umap_model = umap.UMAP(
            n_neighbors=15, 
            metric='cosine', 
            min_dist=0.05, 
            n_components=2, 
            random_state=42
        )
        self.embedding = umap_model.fit_transform(W_dec)

    def compute(self, record, *args, **kwargs):

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


class QuantileSizes(Stat):
    collated = False

    def test_quantiles(
        self,
        record,
        n_reserved=10,
        n_quantiles=4,
        seed=22,
    ):
        random.seed(seed)
        torch.manual_seed(seed)  # Also set torch seed for reproducibility

        remaining_examples = record.examples[n_reserved:]

        # Extract max activations from remaining examples
        max_activations = torch.tensor([example.max_activation for example in remaining_examples])

        # Calculate thresholds based on fractions of the overall maximum activation
        overall_max = max_activations.max().item()
        thresholds = [overall_max * (i + 1) / n_quantiles for i in range(n_quantiles - 1)]
        
        n_per_quantile = []
        start = 0
        for end in thresholds:
            # Filter examples in this quantile
            quantile_examples = [
                example for example in remaining_examples
                if start <= example.max_activation < end
            ]

            n_per_quantile.append(len(quantile_examples))
            
            start = end
        
        # Add the last quantile
        last_quantile = [
            example for example in remaining_examples
            if example.max_activation >= start
        ]

        n_per_quantile.append(len(last_quantile))

        return n_per_quantile

    def compute(self, record, *args, **kwargs):

        n_per_quantile = self.test_quantiles(record)
        record.n_per_quantile = n_per_quantile

class Logits(Stat):
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
        
    def compute(self, records, *args, **kwargs):

        feature_indices = [record.feature.feature_index for record in records]
        
        narrowed_logits = torch.matmul(
            self.W_U, 
            self.W_dec[:,feature_indices]
        )

        top_logits = torch.topk(narrowed_logits, self.k, dim=0).indices

        per_example_top_logits = top_logits.T

        for record_index, record in enumerate(records):

            record.top_logits = \
                self.tokenizer.batch_decode(
                    per_example_top_logits[record_index]
                )


class Activation(Stat):
    collated = False

    def __init__(
        self,
        k=10,
        get_skew=False,
        get_kurtosis=False,
        get_lemmas = False,
        get_similarity=False
    ):
        self.k = k
        
        self.get_skew = get_skew
        self.get_kurtosis = get_kurtosis
        self.sentence_model = None
        self.nlp = None

        if get_similarity:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        if get_lemmas:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")

    def refresh(self, k=None, **kwargs):
        if k:
            self.k = k

    def compute(self, record, *args, **kwargs):

        top_k_examples = record.examples[:self.k]
        top_activations, top_tokens, n_activations =\
            self.top(top_k_examples)
        
        ### PER EXAMPLE STATISTICS ###

        record.average_n_activations = float(np.mean(n_activations))
        record.unique_tokens = len(set(top_tokens))
        
        ### ACTIVATION STATISTICS ###
        
        if self.get_skew:
            record.activation_skew = self.skew(top_activations)
        if self.get_kurtosis:
            record.activation_kurtosis = self.kurtosis(top_activations)
        
        ### ACTIVATION TOKEN STATISTICS ###

        if self.nlp:
            lemmas = self.lemmatize(top_tokens)
            record.lemmas = lemmas
            record.n_lemmas = len(set(lemmas))
        if self.sentence_model:
            record.activation_similarity = self.similarity(
                record.examples[:500]
            ).item()

    def top(self, examples):
        top_activations = []
        top_tokens = []
        n_activations = []
        
        for example in examples:
            # Get top activating token for each example
            max_index = np.argmax(example.activations)

            # Append top activation and token
            top_activations.append(example.activations[max_index].item())
            top_tokens.append(example.str_toks[max_index])
            
            # Count number of activations
            nonzero = np.count_nonzero(example.activations)
            n_activations.append(nonzero)
        
        return top_activations, top_tokens, n_activations
    
    ### ACTIVATION STATISTICS ###

    def kurtosis(self, top_activations):
        return float(kurtosis(top_activations))
    
    def skew(self, top_activations):
        return float(skew(top_activations))
    
    def clean(self, tokens):
        lowercase_tokens = [
            token.lower().strip() 
            for token in tokens
        ]
        alpha_tokens = [
            token for token 
            in lowercase_tokens 
            if token.isalpha()
        ]
        unique_tokens = list(set(alpha_tokens))
        return unique_tokens

    ### ACTIVATION TOKEN STATISTICS ###

    def lemmatize(self, tokens):
        unique_tokens = self.clean(tokens)

        text_for_spacy = " ".join(unique_tokens)

        doc = self.nlp(text_for_spacy)

        lemmatized_tokens = [token.lemma_ for token in doc]
        return lemmatized_tokens

    def similarity(self, examples):
        sentences = [s.text for s in examples]
        embeddings = self.sentence_model.encode(sentences, convert_to_tensor=True)

        # Compute the cosine similarity matrix
        cos_sim = torch.nn.functional.cosine_similarity(
            embeddings.unsqueeze(1), 
            embeddings.unsqueeze(0), 
            dim=2
        )

        # Calculate the average similarity excluding the diagonal
        n = cos_sim.size(0)
        total_similarity = cos_sim.sum() - cos_sim.trace()  # exclude self-similarity
        average_similarity = total_similarity / (n * (n - 1))

        return average_similarity
    


class QuantileActivations(Stat):
    collated = False

    def __init__(
        self,
        k=10,
        get_lemmas = False,
    ):
        self.k = k
        self.nlp = None

        if get_lemmas:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")

    def refresh(self, k=None, **kwargs):
        if k:
            self.k = k

    def compute(self, record, *args, **kwargs):

        top_k_examples = record.examples[:self.k]
        top_activations, top_tokens, n_activations =\
            self.top(top_k_examples)
        
        ### PER EXAMPLE STATISTICS ###

        record.average_n_activations = float(np.mean(n_activations))
        record.unique_tokens = len(set(top_tokens))
        
        ### ACTIVATION TOKEN STATISTICS ###

        if self.nlp:
            lemmas = self.lemmatize(top_tokens)
            record.lemmas = lemmas
            record.n_lemmas = len(set(lemmas))

    def top(self, examples):
        top_activations = []
        top_tokens = []
        n_activations = []
        
        for example in examples:
            # Get top activating token for each example
            max_index = np.argmax(example.activations)

            # Append top activation and token
            top_activations.append(example.activations[max_index].item())
            top_tokens.append(example.str_toks[max_index])
            
            # Count number of activations
            nonzero = np.count_nonzero(example.activations)
            n_activations.append(nonzero)
        
        return top_activations, top_tokens, n_activations
    
    def clean(self, tokens):
        lowercase_tokens = [
            token.lower().strip() 
            for token in tokens
        ]
        alpha_tokens = [
            token for token 
            in lowercase_tokens 
            if token.isalpha()
        ]
        unique_tokens = list(set(alpha_tokens))
        return unique_tokens

    ### ACTIVATION TOKEN STATISTICS ###

    def lemmatize(self, tokens):
        unique_tokens = self.clean(tokens)

        text_for_spacy = " ".join(unique_tokens)

        doc = self.nlp(text_for_spacy)

        lemmatized_tokens = [token.lemma_ for token in doc]
        return lemmatized_tokens