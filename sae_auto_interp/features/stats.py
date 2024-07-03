from typing import List
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from scipy.signal import find_peaks
import torch
from typing import Dict
import torch.nn.functional as F

import umap


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

    def compute(self, records, *args, **kwargs):
        for record in tqdm(records):
            if type(record) == str:
                continue
            for obj in self._objs.values():
                obj.compute(record, *args, **kwargs)


class Neighbors(Stat):

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


class Logits(Stat):

    def __init__(self, 
        model, 
        get_top_logits=False,
        k=10,
        get_skew=False,
        get_kurtosis=False,
        get_entropy=False,
        get_perplexity=False
    ):
        self.model = model
        self.get_top_logits = get_top_logits
        self.k = k
        self.get_skew = get_skew
        self.get_kurtosis = get_kurtosis
        self.get_entropy = get_entropy
        self.get_perplexity = get_perplexity

    def refresh(self, W_dec=None, **kwargs):
        W_U = self.model.transformer.ln_f.weight \
            * self.model.lm_head.weight
        self.logits = torch.matmul(W_U, W_dec).detach().cpu()

    def top_logits(self, logits):
        top_logits = torch.topk(logits, self.k)
        return [
            self.model.tokenizer.decode([token]) 
            for token in top_logits.indices
        ]
    
    def kurtosis(self, logits):
        return float(kurtosis(logits))
    
    def skew(self, logits):
        return float(skew(logits))

    def entropy(self, logits):
        probs = F.softmax(logits, dim=-1)
        return -torch.sum(probs * torch.log(probs)).item()

    def perplexity(self, logits, targets):
        loss = F.cross_entropy(logits.unsqueeze(0), targets.unsqueeze(0), reduction='mean')
        return torch.exp(loss).item()

    def compute(self, record, *args, **kwargs):
        feature_index = record.feature.feature_index
        narrowed_logits = self.logits[feature_index, :]

        if self.get_top_logits:
            record.top_logits = self.top_logits(narrowed_logits)
        if self.get_skew:
            record.logit_skew = self.skew(narrowed_logits)
        if self.get_kurtosis:
            record.logit_kurtosis = self.kurtosis(narrowed_logits)
        if self.get_entropy:
            record.logit_entropy = self.entropy(narrowed_logits)
        if self.get_perplexity:
            # Assuming the target is the token with the highest probability
            target = torch.argmax(narrowed_logits)
            record.logit_perplexity = self.perplexity(narrowed_logits, target)


class Activation(Stat):

    def __init__(
        self,
        k=10,
        get_lemmas = False,
        get_skew=False,
        get_kurtosis=False,
        get_similarity=False,
        n_similar=500
    ):
        self.k = k
        self.get_lemmas = get_lemmas
        self.get_skew = get_skew
        self.get_kurtosis = get_kurtosis
        self.get_similarity = get_similarity
        self.n_similar = n_similar

        if self.get_similarity:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        if self.get_lemmas:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")

    def top(self, examples):
        top_k = examples[:self.k]
        top_activations = []
        top_tokens = []
        n_activations = []
        
        for example in top_k:
            # Get top activating token for each example
            max_index = np.argmax(example.activations)

            # Append top activation and token
            top_activations.append(example.activations[max_index].item())
            top_tokens.append(example.tokens[max_index].item())
            
            # Count number of activations
            nonzero = np.count_nonzero(example.activations)
            n_activations.append(nonzero)
        
        return top_activations, top_tokens, n_activations
    
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
    
    def compute(self, record, *args, **kwargs):
        top_activations, top_tokens, n_activations =\
            self.top(record.examples)
        
        record.top_activations = top_activations
        record.top_tokens = top_tokens
        record.n_activations = n_activations
        record.unique_tokens = len(set(top_tokens))
        
        if self.get_lemmas:
            lemmas = self.lemmatize(top_tokens)
            record.lemmas = lemmas
        if self.get_skew:
            record.activation_skew = self.skew(top_activations)
        if self.get_kurtosis:
            record.activation_kurtosis = self.kurtosis(top_activations)
        if self.get_similarity:
            record.activation_similarity = self.similarity(
                record.examples[:self.n_similar]
            ).item()
