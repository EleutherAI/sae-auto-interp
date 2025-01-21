import os

import torch
from nnsight import LanguageModel
from simple_parsing import ArgumentParser

from sae_auto_interp.autoencoders import load_eai_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data


def main(cfg: CacheConfig, args): 

    size = args.size
    randomize = args.randomize
    k = args.k
    type = args.type

    model = LanguageModel("meta-llama/Meta-Llama-3.1-8B", device_map="auto",dispatch=True,torch_dtype=torch.bfloat16)
    
    if size == "32x":
        weight_dir = "EleutherAI/sae-llama-3.1-8b-32x"
    elif size == "64x":
        weight_dir = "EleutherAI/sae-llama-3.1-8b-64x"
    elif size == "64x_no_multi":
        weight_dir = "/mnt/ssd-1/nora/llama-64x-no-multitopk"

    
    submodule_dict,model = load_eai_autoencoders(
        model,
        [23,29],
        weight_dir,
        module=type,
        randomize=randomize,
        k=k
    )
    

    tokens = load_tokenized_data(
        cfg.ctx_len,
        model.tokenizer,
        cfg.dataset_repo,
        cfg.dataset_split,
    )
    print(submodule_dict)
    cache = FeatureCache(
        model, 
        submodule_dict, 
        batch_size=cfg.batch_size,
    )
    name=""
    if k is not None:
        name += f"_topk_{k}"
    if randomize:
        name += "_random"
    if type != "res":
        name += f"_{type}"
    cache.run(10000000, tokens)
    os.makedirs(f"raw_features/llama/{size}{name}", exist_ok=True)

    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir=f"raw_features/llama/{size}{name}"
    )
    cache.save_config(
        save_dir=f"raw_features/llama/{size}{name}",
        cfg=cfg,
        model_name="meta-llama/Meta-Llama-3.1-8B"
    )

if __name__ == "__main__":

    parser = ArgumentParser()
    #ctx len 256
    parser.add_arguments(CacheConfig, dest="options")
    parser.add_argument("--size", type=str, default="32x")
    parser.add_argument("--type", type=str, default="res")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    cfg = args.options
    print(cfg)
    
    main(cfg, args)
