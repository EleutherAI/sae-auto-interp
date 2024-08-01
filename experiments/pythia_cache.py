from nnsight import LanguageModel
from simple_parsing import ArgumentParser

from sae_auto_interp.autoencoders import load_sam_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_filter, load_tokenized_data


def main(cfg: CacheConfig):
    model = LanguageModel(
        "EleutherAI/pythia-70m-deduped", device_map="auto", dispatch=True
    )
    submodule_dict = load_sam_autoencoders(
        model,
        [4],
        "weights/pythia-70m-deduped",
    )

    module_filter = load_filter("shift/named_sfc.json")

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
        save_dir="/share/u/caden/sae-auto-interp/raw_features/pythia",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options
    main(cfg)
