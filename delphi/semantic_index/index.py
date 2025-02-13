import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from datasets import Dataset
from safetensors.numpy import load_file
from sparsify.data import chunk_and_tokenize
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from delphi.config import CacheConfig
from delphi.latents import LatentDataset
from delphi.logger import logger
from delphi.utils import assert_type
import time


#TODO: Think if this code should be moved to neighbours


class AdversarialContexts:
    """
    Class to compute the neighbours of selected latents using different methods:
    - similarity: uses autoencoder weights
    - correlation: uses pre-activation records and autoencoder
    - co-occurrence: uses latent dataset statistics
    """

    def __init__(
        self,
        
        latent_dataset: Optional['LatentDataset'] = None,
        index_path: Optional[str] = None,
        context_length: int = 32,
        number_of_neighbours: int = 50,
    ):
        
        """
        Initialize a NeighbourCalculator.

        Args:
            latent_dataset (Optional[LatentDataset]): Dataset containing latent activations
            autoencoder (Optional[Autoencoder]): The trained autoencoder model
            residual_stream_record (Optional[ResidualStreamRecord]): Record of residual stream values
        """
        self.latent_dataset = latent_dataset
        self.context_length = context_length
        self.index_path = index_path
        # try to load index
        self.index = self.load_index(index_path)
        
        
    def _compute_similar_contexts(self) -> dict[int, list[int]]:
        """
        Compute neighbour lists based on feature co-occurrence in the dataset.
        If you run out of memory try reducing the token_batch_size
        """

        print("Creating index")
        paths = []
        for buffer in self.latent_dataset.buffers:
            paths.append(buffer.tensor_path)
        
        all_locations = []
        all_activations = []
        tokens = None
        for path in paths:
            split_data = load_file(path)
            first_feature = int(path.split("/")[-1].split("_")[0])
            locations = torch.tensor(split_data["locations"].astype(np.int64))
            locations[:,2] = locations[:,2] + first_feature
            activations = torch.tensor(split_data["activations"].astype(np.float32))
            # compute number of tokens 
            all_locations.append(locations)
            all_activations.append(activations)
            if tokens is None:
                tokens = split_data["tokens"]
        tokens = tokens[:10000]
        reshaped_tokens = tokens.reshape(-1, self.context_length)
        strings = self.latent_dataset.tokenizer.batch_decode(reshaped_tokens, skip_special_tokens=True)
        if self.index is None:
            index = self._build_index(strings)
            self.save_index(index)
        else:
            index = self.index
        
        locations = torch.cat(all_locations)
        activations = torch.cat(all_activations)

        indices = torch.argsort(locations[:,2], stable=True)
        locations = locations[indices]
        activations = activations[indices]
        unique_latents, counts = torch.unique_consecutive(locations[:,2], return_counts=True)
        cache_ctx_len = torch.max(locations[:,1])+1
        
        latents = unique_latents
        split_locations = torch.split(locations, counts.tolist())
        split_activations = torch.split(activations, counts.tolist())
        latents = unique_latents
        dict_of_adversarial_contexts = {}
        for latent, locations, activations in tqdm(zip(latents, split_locations, split_activations)):
            flat_indices = locations[:,0]*cache_ctx_len+locations[:,1]
            ctx_indices = flat_indices // self.context_length
            index_within_ctx = flat_indices % self.context_length
            unique_ctx_indices, inverses, lengths = torch.unique_consecutive(ctx_indices, return_counts=True, return_inverse=True)
            # Get the max activation magnitude within each context window
            max_buffer = torch.segment_reduce(activations, 'max', lengths=lengths)
            k = 100
            _, top_indices = torch.topk(max_buffer, k, sorted=True)
            top_indices = unique_ctx_indices[top_indices]
            # find the context in the index, that are not activating contexts but are the most similar to top_indices
            activating_contexts_indices = ctx_indices.unique()
            print(top_indices.shape)
            print(len(top_indices))
            query_vectors = []
            for i in range(len(top_indices)):
                print(top_indices[i].item())
                breakpoint()
                query_vectors.append(index.reconstruct(top_indices[i].item()))
            
            print(query_vectors.shape)
            distances, indices = index.search(query_vectors, len(top_indices)+len(activating_contexts_indices)+self.number_of_neighbours)
            filtered_indices = []

            for index, distance in zip(indices, distances):
                valid = [id_ for id_ in index if id_ not in activating_contexts_indices]
                filtered_indices.append(valid[:k])
            print(len(filtered_indices))
            dict_of_adversarial_contexts[latent] = filtered_indices
        self.save_adversarial_contexts(dict_of_adversarial_contexts)
                
            

    def _build_index(self,strings: list[str]) -> faiss.IndexIDMap:
        index_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        index_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to("cuda")

        tokenized = index_tokenizer(strings, return_tensors="pt", padding=True,max_length=512,padding_side="right",truncation=True)
        index_tokens = tokenized["input_ids"]
        index_initializer = index_model(index_tokens[:2].to("cuda")).last_hidden_state

        base_index = faiss.IndexFlatL2(index_initializer.shape[-1])
        index = faiss.IndexIDMap(base_index)
        
        
        batch_size = 512
        dataloader = DataLoader(index_tokens, batch_size=batch_size) # type: ignore
        from tqdm import tqdm
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                batch = batch.to("cuda")
                token_embeddings = index_model(batch).last_hidden_state
                sentence_embeddings = token_embeddings.mean(dim=1)
                sentence_embeddings = sentence_embeddings.cpu().numpy().astype(np.float32)
                ids = np.arange(batch_idx * batch_size, batch_idx * batch_size + len(batch))
                index.add_with_ids(sentence_embeddings, ids)
                
        return index

    def populate_neighbour_cache(self, methods: list[str]) -> None:
        """
        Populate the neighbour cache with the computed neighbour lists
        """
        for method in methods:
            self._compute_neighbour_list(method)


    def load_index(self, base_path: str) -> faiss.IndexFlatL2:
        # check if index exists
        index_path = base_path + "/index.faiss"
        if not Path(index_path).exists():
            return None
        return faiss.read_index(str(index_path))

    def save_index(self,index: faiss.IndexFlatL2, ):
        index_path = self.index_path + "/index.faiss"

        faiss.write_index(index, str(index_path))
    
    def save_adversarial_contexts(self, dict_of_adversarial_contexts: dict[int, list[int]]):
        with open(self.index_path + "adversarial_contexts.json", "w") as f:
            json.dump(dict_of_adversarial_contexts, f)

def get_neighbors_by_id(index: faiss.IndexIDMap, vector_id: int, k: int = 10):
    # First reconstruct the vector for the given ID
    vector = index.reconstruct(vector_id)
    
    # Reshape to match FAISS expectations (needs 2D array)
    vector = vector.reshape(1, -1)
    
    # Search for nearest neighbors
    distances, neighbor_ids = index.search(vector, k + 1)  # k+1 since it will find itself
    
    # Remove the first result (which will be the query vector itself)
    return distances[0][1:], neighbor_ids[0][1:]

def get_index_path(base_path: Path, cfg: CacheConfig):
    return base_path / f"{cfg.dataset_repo.replace('/', '_')}_{cfg.dataset_split}_{cfg.ctx_len}.idx"




def build_semantic_index(data: Dataset, cfg: CacheConfig):
        """
        Build a semantic index of the token sequences.
        """
        index_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        index_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to("cuda")

        index_tokens = chunk_and_tokenize(data, index_tokenizer, max_seq_len=cfg.ctx_len, text_key=cfg.dataset_row)
        index_tokens = index_tokens["input_ids"]
        index_tokens = assert_type(Tensor, index_tokens)

        token_embeddings = index_model(index_tokens[:2].to("cuda")).last_hidden_state

        base_index = faiss.IndexFlatL2(token_embeddings.shape[-1])
        index = faiss.IndexIDMap(base_index)

        batch_size = 512
        dataloader = DataLoader(index_tokens, batch_size=batch_size) # type: ignore

        from tqdm import tqdm
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                batch = batch.to("cuda")
                token_embeddings = index_model(batch).last_hidden_state
                sentence_embeddings = token_embeddings.mean(dim=1)
                sentence_embeddings = sentence_embeddings.cpu().numpy().astype(np.float32)

                ids = np.arange(batch_idx * batch_size, batch_idx * batch_size + len(batch))
                index.add_with_ids(sentence_embeddings, ids)

        return index


def build_or_load_index(data: Dataset, base_path: Path, cfg: CacheConfig):
    index_path = get_index_path(base_path, cfg)

    if not index_path.exists():
        logger.info(f"Building semantic index for {cfg.dataset_repo} {cfg.dataset_split} seq_len={cfg.ctx_len}...")
        index = build_semantic_index(data, cfg)
        save_index(index, base_path, cfg)
        return index
    else:
        return load_index(base_path, cfg)
