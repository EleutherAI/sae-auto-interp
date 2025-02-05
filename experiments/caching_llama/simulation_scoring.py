import asyncio
from functools import partial

import orjson
import torch
from simple_parsing import ArgumentParser
import os
from delphi.clients import Local,Outlines
from delphi.config import ExperimentConfig, FeatureConfig
from delphi.explainers import explanation_loader
from delphi.features import FeatureDataset
from delphi.features.constructors import default_constructor
from delphi.features.samplers import sample
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import OpenAISimulator
from delphi.utils import (
    load_tokenized_data,
    load_tokenizer,
)


def main(args):
    module = args.module
    feature_cfg = args.feature_options
    experiment_cfg = args.experiment_options
    batch_size = args.batch_size
    all_at_once = args.all_at_once
    n_features = args.features  
    start_feature = args.start_feature
    sae_model = args.model

    ### Load dataset ###
    #TODO: we should probably save the token information when we save the features that
    # when we load it we don't have to remember all the details.
    tokenizer = load_tokenizer("meta-llama/Meta-Llama-3.1-8B")
    tokens = load_tokenized_data(
        256,
        tokenizer,
        "kh4dien/fineweb-100m-sample",
        "train",
    )
    feature_dict = {f"{module}": torch.arange(start_feature,start_feature+n_features)}
    dataset = FeatureDataset(
        raw_dir=f"raw_features_{sae_model}",
        cfg=feature_cfg,
        modules=[module],
        features=feature_dict,
    )

    #TODO: should be configurable
    
    loader = partial(dataset.load,
        constructor=partial(
            default_constructor,
            tokens=tokens,
            n_random=experiment_cfg.n_random, 
            ctx_len=feature_cfg.example_ctx_len, 
            max_examples=10_000
        ),
        sampler=partial(sample,cfg=experiment_cfg)
    )

    
    client = Outlines("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",base_url=f"http://localhost:{args.port}")
 

    ### Set directories ###

    EXPLAINER_OUT_DIR = f"results/explanations/{sae_model}_{experiment_name}/"
    SCORER_OUT_DIR = f"results/scores/{sae_model}_{experiment_name}_{'all_at_once' if all_at_once else 'token_by_token'}/"

    ### Build Explainer pipe ###

    explainer_pipe = partial(explanation_loader, explanation_dir=EXPLAINER_OUT_DIR)

    ### Build Scorer pipe ###


    def scorer_preprocess(result):
        record = result.record

        record.explanation = result.explanation
        new_test = []
        for i in range(len(record.test)):
            new_test.extend(record.test[i][:2])
        record.test = new_test
        #record.test = record.test[0][:2]
        return record


    def scorer_postprocess(result):
        with open(f"{SCORER_OUT_DIR}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))
    os.makedirs(f"{SCORER_OUT_DIR}", exist_ok=True)


    scorer_pipe = Pipe(
        process_wrapper(
            OpenAISimulator(client, tokenizer=tokenizer, all_at_once=all_at_once),
            preprocess=scorer_preprocess,
            postprocess=scorer_postprocess,
        )
    )

    ### Build the pipeline ###

    pipeline = Pipeline(
        loader,
        explainer_pipe,
        scorer_pipe,
    )

    asyncio.run(
        pipeline.run(max_processes=1)
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--model", type=str, default="128k")
    parser.add_argument("--module", type=str, default=".transformer.h.0")
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--start_feature", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--all_at_once", action='store_true', default=False)
    parser.add_arguments(ExperimentConfig, dest="experiment_options")
    parser.add_arguments(FeatureConfig, dest="feature_options")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    
    experiment_name = args.experiment_name
    


    main(args)
