import json

from nnsight import LanguageModel
import torch
from simple_parsing import ArgumentParser

from sae_auto_interp.config import CacheConfig
from sae_auto_interp.autoencoders import load_oai_autoencoders
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data


def main(cfg: CacheConfig): 
    model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
    submodule_dict = load_oai_autoencoders(
        model, 
        [0,2],
        "weights/gpt2_128k",
    )

    with open("sae_auto_interp/scorers/neighbor/per_layer_features.json") as f:
        data = json.load(f)

    module_filter = {
        name : torch.tensor(data[name], device="cuda:0") 
        for name in list(submodule_dict.keys())
    }

    tokens = load_tokenized_data(model.tokenizer)

    cache = FeatureCache(
        model, 
        submodule_dict, 
        cfg=cfg,
        filters=module_filter
    )
    
    cache.run(cfg.n_tokens, tokens)

    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir="/share/u/caden/sae-auto-interp/temp"
    )

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options
    main(cfg)
