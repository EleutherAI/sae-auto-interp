import asyncio
import json
import os
from functools import partial

import orjson
import torch
import time
from simple_parsing import ArgumentParser

from sae_auto_interp.clients import Offline,OpenRouter
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer
from sae_auto_interp.features import (
    FeatureDataset,
    FeatureLoader
)
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipeline, process_wrapper


def main(args):
    module = args.module
    feature_cfg = args.feature_options
    experiment_cfg = args.experiment_options
    shown_examples = args.shown_examples
    n_features = args.features  
    start_feature = args.start_feature
    sae_model = args.model
    explainer_size = args.explainer_size

    feature_dict = {f"{module}": torch.arange(start_feature,start_feature+n_features)}
    dataset = FeatureDataset(
        raw_dir=f"raw_features/{sae_model}",
        cfg=feature_cfg,
        modules=[module],
        features=feature_dict,
    )

    
    constructor=partial(
            default_constructor,
            tokens=dataset.tokens,
            n_random=experiment_cfg.n_random, 
            ctx_len=experiment_cfg.example_ctx_len, 
            max_examples=feature_cfg.max_examples
        )
    sampler=partial(sample,cfg=experiment_cfg)
    loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)
    ### Load client ###
    if explainer_size == "70b":
        client = Offline("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",max_memory=0.8,max_model_len=5120,num_gpus=1)
    elif explainer_size == "8b":
        client = Offline("meta-llama/Meta-Llama-3.1-8B-Instruct",max_memory=0.8,max_model_len=5120,num_gpus=1)
    elif explainer_size == "claude":
        client = OpenRouter("anthropic/claude-3.5-sonnet",api_key="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")
        
    ### Build Explainer pipe ###
    def explainer_postprocess(result):

        with open(f"results/explanations/{sae_model}/{experiment_name}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.explanation))
        del result
        return None

    #try making the directory if it doesn't exist
    os.makedirs(f"results/explanations/{sae_model}/{experiment_name}", exist_ok=True)

    explainer_pipe = process_wrapper(
        DefaultExplainer(
            client, 
            tokenizer=dataset.tokenizer,
            threshold=0.3,
            activations=True
        ),
        postprocess=explainer_postprocess,
    )

    #save the experiment config
    with open(f"results/explanations/{sae_model}/{experiment_name}/experiment_config.json", "w") as f:
        print(experiment_cfg.to_dict())
        f.write(json.dumps(experiment_cfg.to_dict()))

    ### Build the pipeline ###

    pipeline = Pipeline(
        loader,
        explainer_pipe,
    )
    start_time = time.time()
    asyncio.run(pipeline.run(100))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--shown_examples", type=int, default=5)
    parser.add_argument("--activations", type=bool, default=True)
    parser.add_argument("--explainer_size", type=str, default="70b")
    parser.add_argument("--model", type=str, default="128k")
    parser.add_argument("--module", type=str, default=".transformer.h.0")
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--start_feature", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_arguments(ExperimentConfig, dest="experiment_options")
    parser.add_arguments(FeatureConfig, dest="feature_options")
    args = parser.parse_args()
    
    experiment_name = args.experiment_name
    


    main(args)
