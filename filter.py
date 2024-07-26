# %%

import torch
from functools import partial
from tqdm import tqdm
import torch.multiprocessing as mp

from sae_auto_interp.utils import load_tokenized_data, load_tokenizer
from sae_auto_interp.features import FeatureLoader, FeatureDataset, pool_max_activation_windows
from sae_auto_interp.config import FeatureConfig
from sae_auto_interp.features import unigram

### Set directories ###

RAW_FEATURES_PATH = "raw_features/gpt2"

### Load dataset ###

tokenizer = load_tokenizer('gpt2')
tokens = load_tokenized_data(tokenizer)

# modules = [f".transformer.h.{i}" for i in range(0,12,2)]

modules = [".transformer.h.0"]
features = {
    m : torch.arange(10) for m in modules
}

dataset = FeatureDataset(
    raw_dir=RAW_FEATURES_PATH,
    modules = modules,
    cfg=FeatureConfig(),
    features=features
)

loader = FeatureLoader(
    tokens=tokens,
    dataset=dataset,
    constructor=partial(
        pool_max_activation_windows, 
        ctx_len = 20, 
        max_examples = 3_000
    )
)

def worker(records, k, shared_list, pbar):
    stats = []
    for record in records:
        n_unique, _ = unigram(record, k)
        stats.append(n_unique)
        
    shared_list.append(stats)
    pbar.update(1)

def process_batch(batch, num_processes=4):
    with mp.Manager() as manager:
        shared_list = manager.list()
        progress = tqdm(total=10)

        k_values = range(150, 2150, 200)
        
        with mp.Pool(processes=num_processes) as pool:
            pool.starmap(
                partial(worker, shared_list=shared_list, pbar=progress),
                [(batch, k) for k in k_values]
            )
        
        progress.close()
        
        # Convert the shared list to a regular list
        results = list(shared_list)
    
    return results


# Assuming you have a loader object
for batch in loader.load():
    results = process_batch(batch)

