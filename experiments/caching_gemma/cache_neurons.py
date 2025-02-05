from nnsight import LanguageModel
from simple_parsing import ArgumentParser

from delphi.autoencoders import load_llama3_neurons
from delphi.config import CacheConfig
from delphi.features import FeatureCache
from delphi.utils import load_tokenized_data
import os

def main(cfg: CacheConfig,args): 
    model = LanguageModel("google/gemma-2-9b", device_map="cuda", dispatch=True,torch_dtype="float16")
    layers = args.layers
    k=args.k
    layers = [int(layer) for layer in layers.split(",")]
    
    submodule_dict,model = load_llama3_neurons(
        model,
        layers,
        args.k
    )
    

    tokens = load_tokenized_data(
        cfg.ctx_len,
        model.tokenizer,
        cfg.dataset_repo,
        cfg.dataset_split,
        cfg.dataset_name    
    )

    cache = FeatureCache(
        model, 
        submodule_dict, 
        batch_size=cfg.batch_size,
    )

    cache.run(cfg.n_tokens, tokens)

    if not os.path.exists(f"raw_features/gemma/neurons_{k}"):
        os.makedirs(f"raw_features/gemma/neurons_{k}")

    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir=f"raw_features/gemma/neurons_{k}"
    )

    cache.save_config(
        save_dir=f"raw_features/gemma/neurons_{k}",
        cfg=cfg,
        model_name="google/gemma-2-9b"
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    #ctx len 256
    parser.add_arguments(CacheConfig, dest="options")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--layers", type=str, default="23,27")
    
    args = parser.parse_args()
    cfg = args.options

    main(cfg,args)
