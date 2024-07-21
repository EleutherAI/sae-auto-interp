
# %%

from sae_auto_interp.load.loader import FeatureDataset
from sae_auto_interp.utils import load_tokenized_data
import torch
from nnsight import LanguageModel

import torch.multiprocessing as mp
import time
def main():

    mp.set_start_method('spawn', force=True)
    print("spawned")

    model = LanguageModel('gpt2', device_map="auto")

    tokens = load_tokenized_data(model.tokenizer)

    modules = [".transformer.h.0", ".transformer.h.2"]

    features = {
        m : torch.arange(100) for m in modules
    }

    dataset = FeatureDataset(
        raw_dir="raw_features",
        modules = modules,
        features=features,
        tokens=tokens
    )
    
    for i in dataset.load(n_workers=5):
        print(i)
    
# %%

if __name__ == "__main__":
    start_time = time.time()    
    main()
    print(f"Execution time: {time.time() - start_time}")
