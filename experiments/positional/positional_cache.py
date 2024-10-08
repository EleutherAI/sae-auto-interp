import orjson
import psutil
import torch
from tqdm import tqdm


class FrequencyBuffer:
    def __init__(self, seq_len: int, n_features: int, minibatch_size: int):
        """
        Initialize the cache with the batch length and number of features.

        Args:
            seq_len (int): The length of each sequence in the batch.
            n_features (int): The number of features.
        """
        self.total_counts = torch.zeros(n_features)
        self.position_counts = torch.zeros((seq_len, n_features))
        self.seq_len = seq_len
        self.num_sequences_processed = 0
        self.minibatch_size = minibatch_size

    def final(self):
        """
        Aggregate the counts and calculate the final frequency of
        each feature and each feature at each position.
        """
        fr_n = self.total_counts / (self.num_sequences_processed * self.seq_len)
        fr_n_pos = self.position_counts / self.num_sequences_processed
        return fr_n, fr_n_pos

    def save(self, threshold):
        """
        Finalize the cache and return the sorted indices by mutual information.
        """
        fr_n, fr_n_pos = self.final()
        mutual_information = self.mutual_information_per_feature(
            fr_n_pos, fr_n, self.seq_len
        )
        sorted_indices = self.get_sorted_indices_above_threshold(
            mutual_information, threshold
        )

        return sorted_indices

    def get_sorted_indices_above_threshold(self, mutual_information, threshold):
        """
        Get the indices of features with mutual information above a threshold.
        Threshold specified in [https://arxiv.org/abs/2309.04827]
        """
        indices = torch.where(mutual_information > threshold)[0]
        sorted_indices = indices[
            torch.argsort(mutual_information[indices], descending=True)
        ]

        return sorted_indices

    def update(self, latents):
        """
        Update the cache with the latents from a batch of sequences.
        """
        latents = latents.cpu()
        clamped_latents = torch.where(
            latents > 1e-5, torch.ones_like(latents), torch.zeros_like(latents)
        )
        self.total_counts += torch.sum(clamped_latents, dim=(0, 1))
        self.position_counts += torch.sum(clamped_latents, dim=0)
        self.num_sequences_processed += self.minibatch_size

    def mutual_information_per_feature(self, fr_n_pos: torch.Tensor, fr_n, T):
        """
        Calculate the mutual information between activation frequency and positional frequency.
        From [https://arxiv.org/abs/2309.04827]
        """

        num_positions, num_features = fr_n_pos.shape
        I_act_pos = torch.zeros(num_features)

        for pos in range(num_positions):
            valid_indices = (fr_n_pos[pos] > 0) & (
                fr_n_pos[pos] < 1
            )  # Avoid log(0) or log(infinity)
            if torch.any(valid_indices):
                term1 = fr_n_pos[pos, valid_indices] * torch.log(
                    fr_n_pos[pos, valid_indices] / fr_n[valid_indices]
                )
                term2 = (1 - fr_n_pos[pos, valid_indices]) * torch.log(
                    (1 - fr_n_pos[pos, valid_indices]) / (1 - fr_n[valid_indices])
                )
                I_act_pos[valid_indices] += term1 + term2

        I_act_pos /= T

        return I_act_pos


class FrequencyCache:
    def __init__(
        self,
        model,
        submodule_dict,
        minibatch_size: int,
        seq_len: int,
    ):
        self.model = model
        self.submodule_dict = submodule_dict

        n_features = list(submodule_dict.values())[0].ae.width

        self.seq_len = seq_len
        self.layer_caches = {
            layer: FrequencyBuffer(seq_len, n_features, minibatch_size)
            for layer in submodule_dict.keys()
        }

        self.minibatch_size = minibatch_size

    def check_memory(self, threshold=0.9):
        # Get memory usage as a percentage
        memory_usage = psutil.virtual_memory().percent / 100.0
        return memory_usage > threshold

    def load_token_batches(self, n_tokens, tokens):
        max_batches = n_tokens // self.seq_len
        tokens = tokens[:max_batches]

        n_mini_batches = len(tokens) // self.minibatch_size

        token_batches = [
            tokens[self.minibatch_size * i : self.minibatch_size * (i + 1), :]
            for i in range(n_mini_batches)
        ]

        return token_batches

    def run(self, n_tokens, tokens):
        token_batches = self.load_token_batches(n_tokens, tokens)

        total_tokens = 0
        total_batches = len(token_batches)

        with tqdm(total=total_batches, desc="Caching features") as pbar:
            for batch_number, batch in enumerate(token_batches):
                if self.check_memory(threshold=0.95):
                    print("Memory usage high. Stopping processing.")
                    break

                batch_tokens = batch.numel()
                total_tokens += batch_tokens

                with torch.no_grad():
                    buffer = {}

                    with self.model.trace(batch, scan=False, validate=False):
                        for layer, submodule in self.submodule_dict.items():
                            buffer[layer] = submodule.ae.output.save()

                    for layer, latents in buffer.items():
                        self.layer_caches[layer].update(latents)

                    del buffer
                    torch.cuda.empty_cache()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"Total Tokens": f"{total_tokens:,}"})

        print(f"Total tokens processed: {total_tokens:,}")

    def save(self, threshold=0.05):
        results = {}

        for layer, cache in self.layer_caches.items():
            sorted_indices = cache.save(threshold)
            filename = f"{layer}.txt"

            print(layer, len(sorted_indices))

            with open(filename, "wb") as f:
                f.write(orjson.dumps(sorted_indices.tolist()))

            results[layer] = sorted_indices

        return results
