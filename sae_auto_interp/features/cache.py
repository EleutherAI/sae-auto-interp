import torch
from typing import Tuple, List
from tqdm import tqdm

from datasets import load_dataset, Dataset
from transformer_lens import utils

from .. import cache_cfg

import orjson

import blobfile as bf
import psutil



from collections import defaultdict

N_FEATURES = 32_768

class Buffer:
    def __init__(
        self
    ):
        self.feature_locations = defaultdict(list)
        self.feature_activations = defaultdict(list)
        self.saved = False

    def add(
        self,
        latents: torch.Tensor,
        batch_number: int,
        layer: int
    ):
        feature_locations, feature_activations = self.get_nonzeros(latents)
        feature_locations = feature_locations.cpu()
        feature_activations = feature_activations.cpu()

        feature_locations[:,0] += batch_number * cache_cfg.minibatch_size
        self.feature_locations[layer].append(feature_locations)
        self.feature_activations[layer].append(feature_activations)
        
    def save(self):
        for layer in self.feature_locations.keys():
            self.feature_locations[layer] = torch.cat(self.feature_locations[layer], dim=0)
            self.feature_activations[layer] = torch.cat(self.feature_activations[layer], dim=0)
        
        self.saved = True


    def get_nonzeros(self, latents):
        nonzero_feature_locations = torch.nonzero(latents.abs() > 1e-5)
        nonzero_feature_activations = latents[latents.abs() > 1e-5]

        return nonzero_feature_locations, nonzero_feature_activations
    

    def get_feature_occurances(
        self,
        feature_index: int,
        layer_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if not self.saved:
            self.save()

        locations = self.feature_locations[layer_index]
        activations = self.feature_activations[layer_index]

        # Use boolean indexing
        mask = locations[:, 2] == feature_index
        return locations[mask].tolist(), activations[mask].tolist()

class FeatureCache:

    def __init__(
        self,
        model, 
        ae_dict
    ):  
        self.model = model
        self.ae_dict = ae_dict
        
        self.buffer = Buffer()


    def check_memory(self, threshold=0.9):
        # Get memory usage as a percentage
        memory_usage = psutil.virtual_memory().percent / 100.0
        return memory_usage > threshold


    def load_token_batches(self, minibatch_size=20) -> Tuple[Dataset, List[Tuple[int, int]]]:
        # data = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train[:3%]")
        
        data = load_dataset(cache_cfg.dataset_repo,name=cache_cfg.dataset_name, split=cache_cfg.dataset_split)

        tokens = utils.tokenize_and_concatenate(
            data, 
            self.model.tokenizer, 
            max_length=cache_cfg.batch_len
        )   

        tokens = tokens.shuffle(cache_cfg.seed)['tokens']
        max_batches = cache_cfg.n_tokens // cache_cfg.batch_len
        tokens = tokens[:max_batches]
        
        n_mini_batches = len(tokens) // minibatch_size

        token_batches = [
            tokens[minibatch_size * i : minibatch_size * (i + 1), :] 
            for i in range(n_mini_batches)
        ]

        self.tokens = tokens

        return token_batches
    
    
    def run(self, use_transformer_lens):
        token_batches = self.load_token_batches(cache_cfg.minibatch_size)

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
                    if not use_transformer_lens:
                        with self.model.trace(batch, scan=False, validate=False):
                            for layer, submodule in self.ae_dict.items():
                                buffer[layer] = submodule.ae.output.save()

                        for layer, latents in buffer.items():
                            self.buffer.add(latents, batch_number, layer)
                    else:
                        _, model_acts = self.model.run_with_cache(batch, remove_batch_dim=False)
                        for layer,autoencoder in self.ae_dict.items():
                            layer_acts = model_acts[f"blocks.{layer}.hook_resid_post"]
                            # Get the features of the autoencoder
                            #buffer[layer] = autoencoder(layer_acts)
                            self.buffer.add(autoencoder(layer_acts), batch_number, layer)
                            
                        
                    torch.cuda.empty_cache()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({'Total Tokens': f'{total_tokens:,}'})

        print(f"Total tokens processed: {total_tokens:,}")


    def save_some_features(self, feature_dict):

        self.buffer.save()

        for layer_index, features in feature_dict.items(): 
            layer_index = int(layer_index)
            if layer_index not in [0,2,4,6,8,10]:
                continue
            for feature_index in tqdm(features):
                data = self.buffer.get_feature_occurances(feature_index, layer_index)
                
                self.upload(feature_index, layer_index, data)

    #TODO: I think we want a slightly different save function. 
    def save_features(self):

        if not self.buffer.saved:
            self.buffer.save()

        for layer_index in [0,2,4,6,8,10]:
            for feature_index in tqdm(range(N_FEATURES)):
                data = self.buffer.get_feature_occurances(feature_index, layer_index)
                self.upload(feature_index, layer_index, data)
                
    
    def upload(self, feature_index, layer_index, data):
        with bf.BlobFile(f"gs://gpt2-oai-all/layer{layer_index}_feature{feature_index}", 'wb') as f:
            f.write(
                orjson.dumps(data)
            )
    
   