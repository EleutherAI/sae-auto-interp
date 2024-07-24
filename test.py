
from sae_auto_interp.utils import load_tokenized_data, load_tokenizer, default_constructor
from sae_auto_interp.features import top_and_quantiles, FeatureLoader, FeatureDataset

### Set directories ###

RAW_FEATURES_PATH = "raw_features"
EXPLAINER_OUT_DIR = "results/explanations/simple"
SCORER_OUT_DIR = "results/scores"
SCORER_OUT_DIR_B = "results/scores_b"

### Load dataset ###

def main():
    tokenizer = load_tokenizer('gpt2')
    tokens = load_tokenized_data(tokenizer)

    modules = [".transformer.h.0", ".transformer.h.2"]

    dataset = FeatureDataset(
        raw_dir=RAW_FEATURES_PATH,
        modules = modules,
    )

    loader = FeatureLoader(
        tokens=tokens,
        dataset=dataset,
        constructor=default_constructor,
        sampler=top_and_quantiles
    )

    records = loader.load(collate=True)



if __name__ == "__main__":

    main()
