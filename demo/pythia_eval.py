import asyncio
import orjson

from simple_parsing import ArgumentParser

from functools import partial
from sae_auto_interp.explainers import SimpleExplainer
from sae_auto_interp.scorers import RecallScorer, FuzzingScorer
from sae_auto_interp.clients import Local
from sae_auto_interp.utils import (
    load_tokenized_data,
    load_tokenizer,
)
from defaults import default_constructor
from sae_auto_interp.features import top_and_quantiles, FeatureLoader, FeatureDataset
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.config import FeatureConfig

### Set directories ###

raw_features = "raw_features/pythia"
explanation_dir = "results/pythia_explanations"
recall_dir = "results/pythia_recall"
fuzz_dir = "results/pythia_fuzz"

def main(cfg):
    ### Load dataset ###

    tokenizer = load_tokenizer("gpt2")
    tokens = load_tokenized_data(
        cfg.ctx_len,
        tokenizer,
        "kh4dien/fineweb-100m-sample",
        "train[:15%]",
    )

    dataset = FeatureDataset(
        raw_dir=raw_features,
        cfg=cfg,
    )

    loader = FeatureLoader(
        tokens=tokens,
        dataset=dataset,
        constructor=partial(
            default_constructor, 
            n_random=5, 
            ctx_len=20, 
            max_examples=5_000
        ),
        sampler=top_and_quantiles,
    )

    ### Load client ###

    client = Local("meta-llama/Meta-Llama-3-8B-Instruct")

    ### Build Explainer pipe ###
    def explainer_postprocess(result):
        with open(f"{explanation_dir}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.explanation))
        return result

    explainer_pipe = process_wrapper(
        SimpleExplainer(
            client, 
            tokenizer=tokenizer, 
        ),
        postprocess=explainer_postprocess,
    )

    ### Build Scorer pipe ###

    def scorer_preprocess(result):
        record = result.record
        
        record.explanation = result.explanation
        record.extra_examples = record.random_examples

        return record

    def scorer_postprocess(result, score_dir):
        with open(f"{score_dir}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorer_pipe = Pipe(
        process_wrapper(
            RecallScorer(client, tokenizer=tokenizer, batch_size=cfg.batch_size),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=recall_dir),
        ),
        process_wrapper(
            FuzzingScorer(client, tokenizer=tokenizer, batch_size=cfg.batch_size),
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
