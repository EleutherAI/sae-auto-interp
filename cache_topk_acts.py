from nnsight import LanguageModel
import torch
from simple_parsing import ArgumentParser

from sae_auto_interp.autoencoders import load_gemma_topk_latents
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data


SAVE_DIR = "/mnt/ssd-1/alexm/sae-auto-interp/cache/gemma_topk"


def main(cfg: CacheConfig):
    model_name = "google/gemma-2-9b"
    model = LanguageModel(model_name, device_map="auto", dispatch=True)

    model, submodule_dict = load_gemma_topk_latents(
        model,
        list(range(42)),
        k=50,
    )

    module_filter = {
        module : torch.arange(300).to('cuda:0') 
        for module in submodule_dict
    }

    tokens = load_tokenized_data(
        64,
        model.tokenizer,
        cfg.dataset_repo,
        cfg.dataset_split,
    )

    cache = FeatureCache(
        model, submodule_dict, batch_size=cfg.batch_size, filters=module_filter
    )

    cache.run(cfg.n_tokens, tokens)

    cache.save_splits(
        n_splits=cfg.n_splits,
        save_dir = SAVE_DIR
    )
    cache.save_config(
        save_dir=SAVE_DIR,
        cfg=cfg,
        model_name=model_name,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options

    main(cfg)
