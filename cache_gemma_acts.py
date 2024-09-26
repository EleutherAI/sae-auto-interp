from nnsight import LanguageModel
from simple_parsing import ArgumentParser
import torch
from sae_auto_interp.autoencoders import load_gemma_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.counterfactuals import LAYER_TO_L0
import os


def main(cfg: CacheConfig, args): 
    layers = args.layers
    size = args.size
    type = args.type
    name = args.name
    random = args.random
    model = LanguageModel("google/gemma-2-9b", device_map="cuda", dispatch=True,torch_dtype="bfloat16")
    layers = [int(layer) for layer in layers.split(",")]
    if type == "res":
        dict_l0 = {"131k": LAYER_TO_L0}
    else:
        raise ValueError("Invalid type")
    
    submodule_dict,model = load_gemma_autoencoders(
            model,
            layers,
            {layer: dict_l0[size][layer] for layer in layers},
            size,
            type,
            random
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
    if name not in [""]:
        name = f"_{name}"
    if random:
        name = name + "_random"

    if not os.path.exists(f"raw_features_gemma_{size}{name}"):
        os.makedirs(f"raw_features_gemma_{size}{name}")

    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir=f"raw_features_gemma_{size}{name}"
    )

    cache.save_config(
        save_dir=f"raw_features_gemma_{size}{name}",
        cfg=cfg,
        model_name="google/gemma-2-9b"
    )

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    parser.add_argument("--layers", type=str, default="24,32,41")
    parser.add_argument("--size", type=str, default="131k")
    parser.add_argument("--type", type=str, default="res")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()
    cfg = args.options

    main(cfg,args)
