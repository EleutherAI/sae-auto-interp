import asyncio
from functools import partial

import orjson
from defaults import default_constructor
from simple_parsing import ArgumentParser

from sae_auto_interp.clients import Local
from sae_auto_interp.config import FeatureConfig
from sae_auto_interp.explainers import SimpleExplainer
from sae_auto_interp.features import FeatureDataset, FeatureLoader, top_and_quantiles
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import FuzzingScorer, RecallScorer
from sae_auto_interp.utils import (
    load_tokenized_data,
    load_tokenizer,
)

### Set directories ###

raw_features = "raw_features/pythia"
explanation_dir = "results/pythia_explanations"
recall_dir = "results/pythia_recall"
fuzz_dir = "results/pythia_fuzz"
processed_path = "/share/u/caden/sae-auto-interp/processed_features"


def main(cfg):
    ### Load dataset ###

    tokenizer = load_tokenizer("EleutherAI/pythia-70m-deduped")
    tokens = load_tokenized_data(
        cfg.ctx_len,
        tokenizer,
        "kh4dien/fineweb-100m-sample",
        "train[:15%]",
    )

    dataset = FeatureDataset(
        raw_dir=raw_features,
        cfg=cfg,
        modules=[".gpt_neox.layers.4.attention", ".gpt_neox.layers.4"],
    )

    def load_logits(record):
        try:
            with open(f"{processed_path}/{record.feature}.txt", "rb") as f:
                record.top_logits = orjson.loads(f.read())
        except Exception as e:
            record.top_logits = [f"Top logits could not be loaded ({e})."]
        return record

    loader = FeatureLoader(
        tokens=tokens,
        dataset=dataset,
        constructor=partial(
            default_constructor, n_random=20, ctx_len=20, max_examples=5_000
        ),
        sampler=partial(top_and_quantiles, n_train=20, n_test=7, n_quantiles=10),
        transform=load_logits,
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
            tokenizer=tokenizer,
            cot=True,
            activations=True,
            logits=True,
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
                max_tokens=25,
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
                max_tokens=25,
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

    asyncio.run(pipeline.run())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_arguments(FeatureConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options
    cfg.batch_size = args.batch_size

    main(cfg)
