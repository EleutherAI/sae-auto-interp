import json
from typing import Optional

import numpy as np
import torch
from safetensors.numpy import load_file
from tqdm import tqdm

from delphi.latents import LatentDataset


class NeighbourCalculator:
    """
    Class to compute the neighbours of selected latents using different methods:
    - similarity: uses autoencoder weights
    - correlation: uses pre-activation records and autoencoder
    - co-occurrence: uses latent dataset statistics
    """

    def __init__(
        self,
        latent_dataset: Optional["LatentDataset"] = None,
        autoencoder: Optional["Autoencoder"] = None,
        residual_stream_record: Optional["ResidualStreamRecord"] = None,
        number_of_neighbours: int = 10,
        neighbour_cache: Optional[dict[str, dict[int, list[tuple[int, float]]]]] = None,
    ):
        """
        Initialize a NeighbourCalculator.

        Args:
            latent_dataset (Optional[LatentDataset]): Dataset
                containing latent activations
            autoencoder (Optional[Autoencoder]): The trained autoencoder model
            residual_stream_record (Optional[ResidualStreamRecord]): Record of
                residual stream values
        """
        self.latent_dataset = latent_dataset
        self.autoencoder = autoencoder
        self.residual_stream_record = residual_stream_record
        self.number_of_neighbours = number_of_neighbours

        # load the neighbour cache from the path
        if neighbour_cache is not None:
            self.neighbour_cache = neighbour_cache
        else:
            # dictionary to cache computed neighbour lists
            self.neighbour_cache: dict[str, dict[int, list[int]]] = {}

    def _compute_neighbour_list(self, method: str) -> None:
        """
        Compute complete neighbour lists using specified method.

        Args:
            method (str): One of 'similarity', 'correlation', or 'co-occurrence'
        """
        if method == "similarity_encoder":
            if self.autoencoder is None:
                raise ValueError(
                    "Autoencoder is required for similarity-based neighbours"
                )
            self.neighbour_cache[method] = self._compute_similarity_neighbours(
                "encoder"
            )
        elif method == "similarity_decoder":
            if self.autoencoder is None:
                raise ValueError(
                    "Autoencoder is required for similarity-based neighbours"
                )
            self.neighbour_cache[method] = self._compute_similarity_neighbours(
                "decoder"
            )
        elif method == "correlation":
            if self.autoencoder is None or self.residual_stream_record is None:
                raise ValueError(
                    "Autoencoder and residual stream record are required "
                    "for correlation-based neighbours"
                )
            self.neighbour_cache[method] = self._compute_correlation_neighbours()

        elif method == "co-occurrence":
            if self.latent_dataset is None:
                raise ValueError(
                    "Latent dataset is required for co-occurrence-based neighbours"
                )
            self.neighbour_cache[method] = self._compute_cooccurrence_neighbours()

        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'similarity', 'correlation', "
                "or 'co-occurrence'"
            )

    def _compute_similarity_neighbours(self, method: str) -> dict[int, list[int]]:
        """
        Compute neighbour lists based on weight similarity in the autoencoder.
        """
        print("Computing similarity neighbours")
        # We use the encoder vectors to compute the similarity between latents
        if method == "encoder":
            encoder = self.autoencoder.encoder.cuda()
            weight_matrix_normalized = encoder.weight.data / encoder.weight.data.norm(
                dim=1, keepdim=True
            )

        elif method == "decoder":
            decoder = self.autoencoder.W_dec.cuda()
            weight_matrix_normalized = decoder.data / decoder.data.norm(
                dim=1, keepdim=True
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'encoder' or 'decoder'")

        wT = weight_matrix_normalized.T
        # Compute the similarity between latents
        done = False
        batch_size = weight_matrix_normalized.shape[0]
        number_latents = batch_size

        neighbour_lists = {}
        while not done:
            try:
                for start in tqdm(range(0, number_latents, batch_size)):
                    rows = wT[start : start + batch_size]
                    similarity_matrix = weight_matrix_normalized @ rows
                    indices, values = torch.topk(
                        similarity_matrix, self.number_of_neighbours + 1, dim=1
                    )
                    neighbour_lists.update(
                        {
                            i
                            + start: list(
                                zip(indices[i].tolist()[1:], values[i].tolist()[1:])
                            )
                            for i in range(len(indices))
                        }
                    )
                    del similarity_matrix
                    torch.cuda.empty_cache()
                done = True
            except Exception:
                batch_size = batch_size // 2
                if batch_size < 2:
                    raise ValueError(
                        "Batch size is too small to compute similarity matrix. "
                        "You don't have enough memory."
                    )

        return neighbour_lists

    def _compute_correlation_neighbours(self) -> dict[int, list[int]]:
        """
        Compute neighbour lists based on activation correlation patterns.
        """
        print("Computing correlation neighbours")

        # the activation_matrix has the shape (number_of_samples,hidden_dimension)

        activations = torch.tensor(
            load_file(self.residual_stream_record + ".safetensors")["activations"]
        )

        estimator = CovarianceEstimator(activations.shape[1])
        # batch the activations
        batch_size = 10000
        for i in tqdm(range(0, activations.shape[0], batch_size)):
            estimator.update(activations[i : i + batch_size])

        covariance_matrix = estimator.cov().cuda().half()

        # load the encoder
        encoder_matrix = self.autoencoder.encoder.weight.cuda().half()

        covariance_between_latents = torch.zeros(
            (encoder_matrix.shape[0], encoder_matrix.shape[0]), device="cpu"
        )

        # do batches of latents
        batch_size = 1024
        for start in tqdm(range(0, encoder_matrix.shape[0], batch_size)):
            end = min(encoder_matrix.shape[0], start + batch_size)
            encoder_rows = encoder_matrix[start:end]

            correlation = encoder_rows @ covariance_matrix @ encoder_matrix.T
            covariance_between_latents[start:end] = correlation.cpu()

        # the correlation is then the covariance divided
        # by the product of the standard deviations
        diagonal_covariance = torch.diag(covariance_between_latents)
        product_of_std = torch.sqrt(
            torch.outer(diagonal_covariance, diagonal_covariance) + 1e-6
        )
        correlation_matrix = covariance_between_latents / product_of_std

        # get the indices of the top k neighbours for each feature
        indices, values = torch.topk(
            correlation_matrix, self.number_of_neighbours + 1, dim=1
        )

        # return the neighbour lists
        return {
            i: list(zip(indices[i].tolist()[1:], values[i].tolist()[1:]))
            for i in range(len(indices))
        }

    def _compute_cooccurrence_neighbours(self) -> dict[int, list[int]]:
        """
        Compute neighbour lists based on feature co-occurrence in the dataset.
        If you run out of memory try reducing the token_batch_size
        Code adapted from https://github.com/taha-yassine/SAE-features/blob/main/cooccurrences/compute.py
        """

        import cupy as cp
        import cupyx.scipy.sparse as cusparse

        print("Computing co-occurrence neighbours")
        paths = []
        for buffer in self.latent_dataset.buffers:
            paths.append(buffer.path)

        all_locations = []
        for path in paths:
            split_data = load_file(path)
            first_feature = int(path.split("/")[-1].split("_")[0])
            locations = torch.tensor(split_data["locations"].astype(np.int64))
            locations[:, 2] = locations[:, 2] + first_feature
            # compute number of tokens
            all_locations.append(locations)

        # concatenate the locations and activations
        locations = torch.cat(all_locations)
        n_latents = int(torch.max(locations[:, 2])) + 1

        # 1. Get unique values of first 2 dims (i.e. absolute token index)
        # and their counts
        # Trick is to use Cantor pairing function to have a bijective mapping between
        # (batch_id, ctx_pos) and a unique 1D index
        # Faster than running `torch.unique_consecutive` on the first 2 dims
        idx_cantor = (locations[:, 0] + locations[:, 1]) * (
            locations[:, 0] + locations[:, 1] + 1
        ) // 2 + locations[:, 1]
        unique_idx, idx_counts = torch.unique_consecutive(
            idx_cantor, return_counts=True
        )
        n_tokens = len(unique_idx)

        # 2. The Cantor indices are not consecutive,
        # so we create sorted ones from the counts
        locations_flat = torch.repeat_interleave(
            torch.arange(n_tokens, device=locations.device), idx_counts
        )
        del idx_cantor, unique_idx, idx_counts

        rows = cp.asarray(locations[:, 2])
        cols = cp.asarray(locations_flat)
        data = cp.ones(len(rows))
        sparse_matrix = cusparse.coo_matrix(
            (data, (rows, cols)), shape=(n_latents, n_tokens)
        )
        token_batch_size = 100_000
        cooc_matrix = cp.zeros((n_latents, n_latents), dtype=cp.float32)

        sparse_matrix_csc = sparse_matrix.tocsc()
        for start in tqdm(range(0, n_tokens, token_batch_size)):
            end = min(n_tokens, start + token_batch_size)
            # Slice the sparse matrix to get a batch of tokens.
            sub_matrix = sparse_matrix_csc[:, start:end]
            # Compute the partial co-occurrence matrix for this batch.
            partial_cooc = (sub_matrix @ sub_matrix.T).toarray()
            cooc_matrix += partial_cooc

        # Free temporary variables.
        del rows, cols, data, sparse_matrix, sparse_matrix_csc

        # Compute Jaccard similarity
        def compute_jaccard(cooc_matrix):
            self_occurrence = cooc_matrix.diagonal()
            jaccard_matrix = cooc_matrix / (
                self_occurrence[:, None] + self_occurrence - cooc_matrix
            )
            # remove the diagonal and keep the upper triangle
            return jaccard_matrix

        # Compute Jaccard similarity matrix
        jaccard_matrix = compute_jaccard(cooc_matrix)

        jaccard_torch = torch.as_tensor(cp.asnumpy(jaccard_matrix))
        # get the indices of the top k neighbours for each feature
        top_k_indices, values = torch.topk(
            jaccard_torch, self.number_of_neighbours + 1, dim=1
        )
        del jaccard_matrix, cooc_matrix, jaccard_torch
        torch.cuda.empty_cache()

        # return the neighbour lists
        return {
            i: list(zip(top_k_indices[i].tolist()[1:], values[i].tolist()[1:]))
            for i in range(len(top_k_indices))
        }

    def populate_neighbour_cache(self, methods: list[str]) -> None:
        """
        Populate the neighbour cache with the computed neighbour lists
        """
        for method in methods:
            self._compute_neighbour_list(method)

    def save_neighbour_cache(self, path: str) -> None:
        """
        Save the neighbour cache to the path as a json file
        """
        with open(path, "w") as f:
            json.dump(self.neighbour_cache, f)

    def load_neighbour_cache(self, path: str) -> dict[str, dict[int, list[int]]]:
        """
        Load the neighbour cache from the path as a json file
        """
        with open(path, "r") as f:
            return json.load(f)


class CovarianceEstimator:
    def __init__(self, n_latents, *, device=None):
        self.mean = torch.zeros(n_latents, device=device)
        self.cov_ = torch.zeros(n_latents, n_latents, device=device)
        self.n = 0

    def update(self, x: torch.Tensor):
        n, d = x.shape
        assert d == len(self.mean)

        self.n += n

        # Welford's online algorithm
        delta = x - self.mean
        self.mean.add_(delta.sum(dim=0), alpha=1 / self.n)
        delta2 = x - self.mean

        self.cov_.addmm_(delta.mH, delta2)

    def cov(self):
        """Return the estimated covariance matrix."""
        return self.cov_ / self.n
