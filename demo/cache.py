from nnsight import LanguageModel
import torch
from simple_parsing import ArgumentParser

from sae_auto_interp.autoencoders import load_oai_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data


WEIGHT_PATH = "weights/gpt2_128k"
SAVE_DIR = "/share/u/caden/sae-auto-interp/raw_features/new_stuff"


def main(cfg: CacheConfig):
    model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

    submodule_dict = load_oai_autoencoders(
        model,
        list(range(0, 12, 2)),
        WEIGHT_PATH
    )

    module_filter = {
        module : torch.arange(100).to('cuda:0') 
        for module in submodule_dict
    }

    tokens = load_tokenized_data(
        cfg.ctx_len,
        model.tokenizer,
        "kh4dien/fineweb-100m-sample",
        "train[:15%]",
    )

    cache = FeatureCache(
        model, submodule_dict, batch_size=cfg.batch_size, filters=module_filter
    )

    cache.run(cfg.n_tokens, tokens)

    cache.save_splits(
        n_splits=cfg.n_splits,
        save_dir = SAVE_DIR
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options

    main(cfg)
