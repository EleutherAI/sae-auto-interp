import asyncio
import os
from functools import partial

import orjson
import torch
from simple_parsing import ArgumentParser
from transformers import AutoTokenizer

from sae_auto_interp.clients import Offline
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import explanation_loader
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import OpenAISimulator


def main(args):
    module = args.module
    feature_cfg = args.feature_options
    experiment_cfg = args.experiment_options
    all_at_once = args.all_at_once
    n_features = args.features  
    start_feature = args.start_feature
    sae_model = args.model
    scorer_size = args.scorer_size

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
    

    
    #client = Local("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",base_url=f"http://localhost:{args.port}/v1")
    if scorer_size == "70b":
        client = Offline("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",max_memory=0.5, max_model_len=5120,num_gpus=2,batch_size=5,prefix_caching=False)
        SCORER_OUT_DIR = f"results/scores/{sae_model}/{experiment_name}"
    elif scorer_size == "8b":
        client = Offline("meta-llama/Meta-Llama-3.1-8B-Instruct",max_memory=0.5,max_model_len=5120,num_gpus=1,batch_size=5,prefix_caching=False)
        SCORER_OUT_DIR = f"results/scores/{sae_model}/{experiment_name}_scorer_8b"
    elif scorer_size == "claude":
        client = OpenRouter("anthropic/claude-3.5-sonnet",api_key="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")
        SCORER_OUT_DIR = f"results/scores/{sae_model}/{experiment_name}_scorer_claude"
    
    ### Set directories ###

    EXPLAINER_OUT_DIR = f"results/explanations/{sae_model}/{experiment_name}/"
    
    ### Build Explainer pipe ###

    explainer_pipe = partial(explanation_loader, explanation_dir=EXPLAINER_OUT_DIR)

    ### Build Scorer pipe ###


    def scorer_preprocess(result):
        record = result.record
        record.explanation = result.explanation
        new_test = []
        for i in range(len(record.test)):
            new_test.append(record.test[i])
        record.test = new_test
        #record.test = record.test[0][:2]
        return record


    def scorer_postprocess(result):
        with open(f"{SCORER_OUT_DIR}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))
    os.makedirs(f"{SCORER_OUT_DIR}", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
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
        pipeline.run(10)
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--model", type=str, default="128k")
    parser.add_argument("--module", type=str, default=".transformer.h.0")
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--start_feature", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--no_all_at_once", action='store_false', dest="all_at_once")
    parser.add_argument("--scorer_size", type=str, default="70b")
    parser.add_arguments(ExperimentConfig, dest="experiment_options")
    parser.add_arguments(FeatureConfig, dest="feature_options")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    
    experiment_name = args.experiment_name
    


    main(args)
