import asyncio
from functools import partial

import orjson
import torch
from defaults import default_constructor
from simple_parsing import ArgumentParser

from sae_auto_interp.clients import Local
from sae_auto_interp.config import FeatureConfig
from sae_auto_interp.explainers import SimpleExplainer
from sae_auto_interp.features import FeatureDataset, FeatureLoader, random_and_quantiles
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import FuzzingScorer, RecallScorer
from sae_auto_interp.utils import (
    load_tokenized_data,
    load_tokenizer,
)

### Set directories ###

raw_features = "raw_features/gpt2"
explanation_dir = "results/gpt2_explanations"
recall_dir = "results/gpt2_recall"
fuzz_dir = "results/gpt2_fuzz"
processed_path = "/share/u/caden/sae-auto-interp/processed_features"


def main(cfg):
    ### Load dataset ###

    tokenizer = load_tokenizer("gpt2")
    tokens = load_tokenized_data(
        cfg.ctx_len,
        tokenizer,
        "kh4dien/fineweb-100m-sample",
        "train[:15%]",
    )

    modules = [f".transformer.h.{i}" for i in range(0, 12, 2)]

    features = {mod: torch.arange(50) for mod in modules}

    dataset = FeatureDataset(
        raw_dir=raw_features,
        cfg=cfg,
        modules=modules,
        features=features,
    )

    loader = FeatureLoader(
        tokens=tokens,
        dataset=dataset,
        constructor=partial(
            default_constructor, n_random=20, ctx_len=20, max_examples=5_000
        ),
        sampler=partial(random_and_quantiles, n_train=20, n_test=7, n_quantiles=10),
    )

    ### Load client ###

    client = Local("casperhansen/llama-3-70b-instruct-awq")

    ### Build Explainer pipe ###

    def preprocess(record):
        test = []
        extra_examples = []

        for examples in record.test:
            test.append(examples[:5])
            extra_examples.extend(examples[5:])

        record.test = test
        record.extra_examples = extra_examples

        return record

    def explainer_postprocess(result):
        data = {
            "generation_prompt": result[0],
            "response": result[1],
            "explanation": result[2].explanation,
        }

        with open(f"{explanation_dir}/{result[2].record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(data))

        return result[2]

    explainer_pipe = process_wrapper(
        SimpleExplainer(
            client,
            verbose=True,
            tokenizer=tokenizer,
            activations=True,
            max_tokens=500,
            temperature=0.0,
        ),
        preprocess=preprocess,
        postprocess=explainer_postprocess,
    )

    ### Build Scorer pipe ###

    def scorer_preprocess(result):
        record = result.record

        record.explanation = result.explanation

        return record

    def scorer_postprocess(result, score_dir):
        with open(f"{score_dir}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorer_pipe = Pipe(
        process_wrapper(
            RecallScorer(
                client,
                tokenizer=tokenizer,
                verbose=True,
                max_tokens=50,
                temperature=0.0,
                batch_size=cfg.batch_size,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=recall_dir),
        ),
        process_wrapper(
            FuzzingScorer(
                client,
                tokenizer=tokenizer,
                verbose=True,
                max_tokens=50,
                temperature=0.0,
                batch_size=cfg.batch_size,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=fuzz_dir),
        ),
    )

    ### Build the pipeline ###

    pipeline = Pipeline(
        loader.load,
        explainer_pipe,
        scorer_pipe,
    )

    asyncio.run(pipeline.run(max_processes=5))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_arguments(FeatureConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options
    cfg.batch_size = args.batch_size

    main(cfg)
