import os

import torch
from nnsight import LanguageModel
from simple_parsing import ArgumentParser

from sae_auto_interp.autoencoders import load_eai_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data


def main(cfg: CacheConfig, args): 

    skip = args.skip
    transcoder = args.transcoder
    
    model = LanguageModel("EleutherAI/pythia-160m", device_map="auto",dispatch=True,torch_dtype=torch.bfloat16)
    
    weight_dir = "/mnt/ssd-1/nora/sae/k32-skip-32k"
    
    submodule_dict,model = load_eai_autoencoders(
        model,
        [0,1,2,3,4,5,6,7,8,9,10,11],
        weight_dir,
        module="mlp",
        transcoder=transcoder
    )

    tokens = load_tokenized_data(
        cfg.ctx_len,
        model.tokenizer,
        cfg.dataset_repo,
        cfg.dataset_split,
        cfg.dataset_name,
        cfg.dataset_column_name,
    )
    print(submodule_dict)
    cache = FeatureCache(
        model, 
        submodule_dict, 
        batch_size=cfg.batch_size,
    )
    

    cache.run(cfg.n_tokens, tokens)
    os.makedirs("raw_features/transcoder", exist_ok=True)
    print(f"Saving splits to {'raw_features/transcoder'}")
    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir="raw_features/transcoder"
    )
    print(f"Saving config to {'raw_features/transcoder'}")
    cache.save_config(
        save_dir="raw_features/transcoder",
        cfg=cfg,
        model_name="EleutherAI/pythia-160m"
    )

if __name__ == "__main__":

    parser = ArgumentParser()
    #ctx len 256
    parser.add_arguments(CacheConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options
    print(cfg)
    
    main(cfg, args)
