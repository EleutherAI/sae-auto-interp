from nnsight import LanguageModel
from simple_parsing import ArgumentParser

from sae_auto_interp.autoencoders import load_llama3_neurons
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data
import os

def main(cfg: CacheConfig,args): 
    model = LanguageModel("meta-llama/Meta-Llama-3.1-8B", device_map="auto", dispatch=True)
    
    submodule_dict,model = load_llama3_neurons(
        model,
        [23,29],
        args.k
    )
    

    tokens = load_tokenized_data(
        cfg.ctx_len,
        model.tokenizer,
        cfg.dataset_repo,
        cfg.dataset_split,
    )

    cache = FeatureCache(
        model, 
        submodule_dict, 
        batch_size=cfg.batch_size,
    )

    cache.run(10000000, tokens)
    os.makedirs(f"raw_features/llama/neurons_{args.k}", exist_ok=True)

    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir=f"raw_features/llama/neurons_{args.k}"
    )
    cache.save_config(
        save_dir=f"raw_features/llama/neurons_{args.k}",
        cfg=cfg,
        model_name="meta-llama/Meta-Llama-3.1-8B"
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    #ctx len 256
    parser.add_arguments(CacheConfig, dest="options")
    parser.add_argument("--k", type=int, default=32)
    args = parser.parse_args()
    cfg = args.options

    main(cfg,args)
