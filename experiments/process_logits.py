import orjson
from nnsight import LanguageModel
from simple_parsing import ArgumentParser

from sae_auto_interp.autoencoders import load_sam_autoencoders
from sae_auto_interp.config import FeatureConfig
from sae_auto_interp.features import (
    FeatureDataset,
    FeatureLoader,
)
from sae_auto_interp.features.stats import logits
from sae_auto_interp.utils import load_tokenized_data, load_tokenizer

### Set directories ###

RAW_FEATURES_PATH = "raw_features/pythia"
EXPLAINER_OUT_DIR = "results/explanations"
SCORER_OUT_DIR = "results/generation"


def main(cfg):
    ### Load dataset ###
    model = LanguageModel(
        "EleutherAI/pythia-70m-deduped", device_map="auto", dispatch=True
    )
    submodule_dict = load_sam_autoencoders(
        model,
        list(range(5)),
        "weights/pythia-70m-deduped",
    )

    tokenizer = load_tokenizer("EleutherAI/pythia-70m-deduped")
    tokens = load_tokenized_data(
        cfg.ctx_len,
        tokenizer,
        "kh4dien/fineweb-100m-sample",
        "train[:15%]",
    )

    dataset = FeatureDataset(
        raw_dir=RAW_FEATURES_PATH,
        cfg=cfg,
    )

    loader = FeatureLoader(
        tokens=tokens,
        dataset=dataset,
    )

    for batch in loader.load():
        module = batch[0].feature.module_name

        ae = submodule_dict[module].ae

        W_U = model.gpt_neox.final_layer_norm.weight * model.embed_out.weight

        W_dec = ae.ae._module.decoder.weight

        logits(batch, W_U=W_U, W_dec=W_dec, k=10, tokenizer=tokenizer)

        for record in batch:
            with open(f"processed_features/{record.feature}.txt", "wb") as f:
                f.write(orjson.dumps(record.top_logits))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(FeatureConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options
    main(cfg)
