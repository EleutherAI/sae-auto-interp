import asyncio
from functools import partial

import orjson
import torch
from simple_parsing import ArgumentParser

from sae_auto_interp.clients import Outlines
from sae_auto_interp.config import FeatureConfig
from sae_auto_interp.explainers import explanation_loader
from sae_auto_interp.features import (
    FeatureDataset,
    FeatureLoader,
)
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.scorers import GenerationScorer
from sae_auto_interp.utils import load_tokenized_data, load_tokenizer

### Set directories ###

RAW_FEATURES_PATH = "raw_features/gpt2"
EXPLAINER_OUT_DIR = "results/explanations"
SCORER_OUT_DIR = "results/generation"


def main(cfg):
    ### Load dataset ###

    tokenizer = load_tokenizer("gpt2")
    tokens = load_tokenized_data(
        cfg.seq_len,
        tokenizer,
        "kh4dien/fineweb-100m-sample",
        "train[:15%]",
    )

    modules = [f".transformer.h.{i}" for i in range(0, 12, 2)]
    features = {m: torch.arange(10) for m in modules}

    dataset = FeatureDataset(
        raw_dir=RAW_FEATURES_PATH,
        modules=modules,
        cfg=cfg,
        features=features,
    )

    loader = FeatureLoader(
        tokens=tokens,
        dataset=dataset,
        # constructor=partial(
        #     pool_max_activation_windows,
        #     ctx_len=cfg.ctx_len,
        #     max_examples=cfg.max_examples,
        # ),
    )

    ### Load client ###

    client = Outlines("meta-llama/Meta-Llama-3-8B-Instruct")

    ### Build Explainer pipe ###

    explainer = partial(explanation_loader, explanation_dir=EXPLAINER_OUT_DIR)

    ### Build Scorer pipe ###

    def scorer_preprocess(result):
        record = result.record
        record.explanation = result.explanation
        return record

    def scorer_postprocess(result):
        with open(f"{SCORER_OUT_DIR}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorer_pipe = process_wrapper(
        GenerationScorer(client, max_tokens=1500, temperature=0.5),
        preprocess=scorer_preprocess,
        postprocess=scorer_postprocess,
    )

    ### Build the pipeline ###

    pipeline = Pipeline(
        loader.load,
        explainer,
        scorer_pipe,
    )

    asyncio.run(pipeline.run())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(FeatureConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options

    main(cfg)
