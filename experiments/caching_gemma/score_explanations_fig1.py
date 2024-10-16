import asyncio
import os
from functools import partial
from typing import NamedTuple

import orjson
import torch
from simple_parsing import ArgumentParser

from sae_auto_interp.clients import Offline, OpenRouter
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import FuzzingScorer, DetectionScorer
import aiofiles
import json
from sae_auto_interp.features import FeatureRecord

class ExplainerResult(NamedTuple):
    record: FeatureRecord
    """Feature record passed through to scorer."""

    explanation: str
    """Generated explanation for feature."""


async def explanation_loader(record: FeatureRecord, explanation_dir: str) -> ExplainerResult:
    feature = str(record.feature)
    layer = feature.split("_feature")[0].split(".")[-1]
    feature = feature.split("_feature")[1].split(".")[0]
    async with aiofiles.open(f'extras/explanations_131k/model.layers.{layer}_feature.json', 'r') as f:
        explanations = json.loads(await f.read())
    explanation = explanations[feature]
    print()
    print(explanation)
    print(record.feature)
    return ExplainerResult(
        record=record,
        explanation=explanation
    )



def main(args):
    feature_cfg = args.feature_options
    experiment_cfg = ExperimentConfig()
    experiment_name = "fig1"
    
    feature_dict = {#".model.layers.0":torch.tensor([44,2528,2011]),
                    #".model.layers.5":torch.tensor([3031,8603]),
                    #".model.layers.15":torch.tensor([6680]),
                    #".model.layers.20":torch.tensor([7343]),
                    ".model.layers.40":torch.tensor([4661]),
                    }
    dataset = FeatureDataset(
        raw_dir="raw_features/gemma/131k",
        cfg=feature_cfg,
        modules=[".model.layers.40"],
        features=feature_dict,
    )
    
    constructor=partial(
            default_constructor,
            tokens=dataset.tokens,
            n_random=100, 
            ctx_len=32, 
            max_examples=10000
        )
    experiment_cfg.n_examples_test=10
    experiment_cfg.n_quantiles=10
    experiment_cfg.example_ctx_len=32
    sampler=partial(sample,cfg=experiment_cfg)
    loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)
    ### Load client ###

    EXPLAINER_OUT_DIR = f"results/explanations/gemma/131k/{experiment_name}"

    ### Build Explainer pipe ###
    explainer_pipe = partial(explanation_loader, explanation_dir=EXPLAINER_OUT_DIR)


    client = Offline("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",max_memory=0.8, max_model_len=5120,num_gpus=2)
    SCORE_OUT_DIR = f"results/scores/gemma/131k/{experiment_name}"

    
    ### Build Scorer pipe ###

    def scorer_preprocess(result):
        record = result.record
        
        record.explanation = result.explanation
        record.extra_examples = record.random_examples

        return record

    def scorer_postprocess(result, score_dir):
        with open(f"{SCORE_OUT_DIR}/{score_dir}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    os.makedirs(f"{SCORE_OUT_DIR}/detection", exist_ok=True)
    os.makedirs(f"{SCORE_OUT_DIR}/fuzz", exist_ok=True)
    scorer_pipe = Pipe(
        process_wrapper(
            DetectionScorer(client, tokenizer=dataset.tokenizer, batch_size=5,verbose=False,log_prob=True),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="detection"),
        ),
        process_wrapper(
            FuzzingScorer(client, tokenizer=dataset.tokenizer, batch_size=5,verbose=False,log_prob=True),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="fuzz"),
        ),
    )

    ### Build the pipeline ###

    pipeline = Pipeline(
        loader,
        explainer_pipe,
        scorer_pipe,
    )

    asyncio.run(pipeline.run(50))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(FeatureConfig, dest="feature_options")
    args = parser.parse_args()


    main(args)
