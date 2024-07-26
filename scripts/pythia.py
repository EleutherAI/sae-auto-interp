import json

from nnsight import LanguageModel
from simple_parsing import ArgumentParser

from sae_auto_interp.config import CacheConfig
from sae_auto_interp.autoencoders import load_sam_autoencoders
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data, load_filter


def main(cfg: CacheConfig): 
    model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map="auto", dispatch=True)
    submodule_dict = load_sam_autoencoders(
        model, 
        list(range(4)),
        "weights/pythia-70m-deduped",
    )

    with open("shift/named_sfc.json") as f:
        module_filter = load_filter(json.load(f))

    tokens = load_tokenized_data(model.tokenizer)

    cache = FeatureCache(
        model, 
        submodule_dict, 
        cfg = cfg,
        filters = module_filter
    )
    
    cache.run(cfg.n_tokens, tokens)

    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir="/share/u/caden/sae-auto-interp/raw_features/pythia"
    )

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options
    main(cfg)
