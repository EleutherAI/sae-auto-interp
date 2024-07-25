import asyncio
import orjson

import argparse

import torch
import time
from sae_auto_interp.explainers import SimpleExplainer
from sae_auto_interp.scorers import RecallScorer
from sae_auto_interp.clients import Local
from sae_auto_interp.utils import load_tokenized_data, load_tokenizer, default_constructor
from sae_auto_interp.features import top_and_quantiles, FeatureLoader, FeatureDataset
from sae_auto_interp.sequential_pipeline import Pipe, Pipeline, Actor
from sae_auto_interp.config import FeatureConfig

### Set directories ###

RAW_FEATURES_PATH = "raw_features/gpt2"
EXPLAINER_OUT_DIR = "results/explanations"
SCORER_OUT_DIR = "results/scores"

def main(batch_size: int):

    ### Load dataset ###

    tokenizer = load_tokenizer('gpt2')
    tokens = load_tokenized_data(tokenizer)

    modules = [f".transformer.h.{i}" for i in range(0,12,2)]
    features = {
        m : torch.arange(100) for m in modules
    }

    dataset = FeatureDataset(
        raw_dir=RAW_FEATURES_PATH,
        modules = modules,
        cfg=FeatureConfig(),
        features=features,
    )

    loader = FeatureLoader(
        tokens=tokens,
        dataset=dataset,
        constructor=default_constructor,
        sampler=top_and_quantiles
    )

    ### Load client ###

    client = Local("meta-llama/Meta-Llama-3-8B-Instruct")

    ### Build Explainer pipe ###

    def explainer_preprocess(record):
        record.time = time.time()

        return record

    def explainer_postprocess(result):
        result = result.result()

        data= {
            "time" : time.time() - result.record.time,
            "explanation": result.explanation
        }

        with open(f"{EXPLAINER_OUT_DIR}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(data))

    explainer_pipe = Pipe(
        Actor(
            SimpleExplainer(client, tokenizer=tokenizer, cot=True),
            preprocess=explainer_preprocess,
            postprocess=explainer_postprocess
        ),
        name="explainer"
    )

    ### Build Scorer pipe ###

    def scorer_preprocess(result):

        record = result.record
        record.time = time.time()
        record.explanation = result.explanation

        return record

    def scorer_postprocess(result):
        result = result.result()

        data= {
            "time" : time.time() - result.record.time,
            "score": result.score
        }

        with open(f"{SCORER_OUT_DIR}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(data))

    scorer_pipe = Pipe(
        Actor(
            RecallScorer(client, tokenizer=tokenizer, batch_size=batch_size),
            preprocess=scorer_preprocess,
            postprocess=scorer_postprocess
        ),
        name="scorer"
    )

    ### Build the pipeline ###

    pipeline = Pipeline(
        loader.load,
        explainer_pipe,
        # scorer_pipe,
    )

    asyncio.run(
        pipeline.run()
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the recall pipeline')
    parser.add_argument('--batch_size', type=int, help='Batch size of scorer')

    args = parser.parse_args()

    main(args.batch_size)