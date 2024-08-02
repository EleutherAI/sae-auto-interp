import asyncio
import os
from functools import partial

import orjson
import torch
from simple_parsing import ArgumentParser

from sae_auto_interp.clients import Local
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import SimpleExplainer
from sae_auto_interp.features import (
    FeatureDataset,
)
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import FuzzingScorer, RecallScorer
from sae_auto_interp.utils import (
    load_tokenized_data,
    load_tokenizer,
)

### Set directories ###


def main(args):
    module = args.module
    feature_cfg = args.feature_options
    experiment_cfg = args.experiment_options
    #TODO: this should not be called batch_size
    batch_size = args.batch_size
    n_features = args.features  
    start_feature = args.start_feature
    sae_model = args.model

    ### Load dataset ###
    #TODO: we should probably save the token information when we save the features that
    # when we load it we don't have to remember all the details.
    tokenizer = load_tokenizer("meta-llama/Meta-Llama-3-8B")
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
            max_examples=5_000
        ),
        sampler=partial(sample,cfg=experiment_cfg)
    )

    ### Load client ###
    # if sae_model=="llama_neurons_3":
    #     client = Local("casperhansen/llama-3-70b-instruct-awq",base_url="http://localhost:8001/v1")
    # else:
    #    
    client = Local("casperhansen/llama-3-70b-instruct-awq",base_url=f"http://localhost:{args.port}/v1")
 

    ### Build Explainer pipe ###
    def explainer_postprocess(result):

        with open(f"results/explanations/{sae_model}_{experiment_name}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.explanation))

        return result
    #try making the directory if it doesn't exist
    os.makedirs(f"results/explanations/{sae_model}_{experiment_name}", exist_ok=True)

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
        with open(f"results/scores/{sae_model}_{experiment_name}_{score_dir}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    os.makedirs(f"results/scores/{sae_model}_{experiment_name}_recall", exist_ok=True)
    os.makedirs(f"results/scores/{sae_model}_{experiment_name}_fuzz", exist_ok=True)

    scorer_pipe = Pipe(
        process_wrapper(
            RecallScorer(client, tokenizer=tokenizer, batch_size=batch_size,verbose=False,log_prob=True),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="recall"),
        ),
        process_wrapper(
            FuzzingScorer(client, tokenizer=tokenizer, batch_size=batch_size,verbose=False,log_prob=True),
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

    asyncio.run(pipeline.run(max_processes=20))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--model", type=str, default="128k")
    parser.add_argument("--module", type=str, default=".transformer.h.0")
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--start_feature", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_arguments(ExperimentConfig, dest="experiment_options")
    parser.add_arguments(FeatureConfig, dest="feature_options")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    
    experiment_name = args.experiment_name
    


    main(args)
