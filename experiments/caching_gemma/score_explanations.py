import asyncio
import os
from functools import partial

import orjson
import torch
from simple_parsing import ArgumentParser

from delphi.clients import Offline,OpenRouter
from delphi.config import ExperimentConfig, FeatureConfig
from delphi.explainers import  explanation_loader,random_explanation_loader
from delphi.features import (
    FeatureDataset,
    FeatureLoader
)
from delphi.features.constructors import default_constructor
from delphi.features.samplers import sample
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import FuzzingScorer, DetectionScorer


def main(args):
    module = args.module
    feature_cfg = args.feature_options
    experiment_cfg = args.experiment_options
    shown_examples = args.shown_examples
    n_features = args.features  
    start_feature = args.start_feature
    sae_model = args.model
    experiment_name = args.experiment_name
    quantization = args.quantization
    scorer_size = args.scorer_size
    log_prob = args.prob

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

    EXPLAINER_OUT_DIR = f"results/explanations/{sae_model}/{experiment_name}"

    ### Build Explainer pipe ###
    if args.random:
        explainer_pipe = partial(random_explanation_loader, explanation_dir=EXPLAINER_OUT_DIR)
        experiment_name = "random_explanation"
    else:
        explainer_pipe = partial(explanation_loader, explanation_dir=EXPLAINER_OUT_DIR)


    if scorer_size == "70b":
        client = Offline("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",max_memory=0.8, max_model_len=5120,num_gpus=2)
        SCORE_OUT_DIR = f"results/scores/{sae_model}/{experiment_name}"
    elif scorer_size == "8b":
        client = Offline("meta-llama/Meta-Llama-3.1-8B-Instruct",max_memory=0.8,max_model_len=5120,num_gpus=1)
        SCORE_OUT_DIR = f"results/scores/{sae_model}/{experiment_name}_scorer_8b"
    elif scorer_size == "claude":
        client = OpenRouter("anthropic/claude-3.5-sonnet",api_key=os.getenv("OPENROUTER_API_KEY"))
        SCORE_OUT_DIR = f"results/scores/{sae_model}/{experiment_name}_scorer_claude"
    
    
    ### Build Scorer pipe ###

    def scorer_preprocess(result):
        record = result.record
        
        record.explanation = result.explanation
        record.extra_examples = record.negative_examples

        return record

    def scorer_postprocess(result, score_dir):
        with open(f"{SCORE_OUT_DIR}/{score_dir}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    os.makedirs(f"{SCORE_OUT_DIR}/detection", exist_ok=True)
    os.makedirs(f"{SCORE_OUT_DIR}/fuzz", exist_ok=True)
    scorer_pipe = Pipe(
        process_wrapper(
            detectionScorer(client, tokenizer=dataset.tokenizer, batch_size=shown_examples,verbose=False,log_prob=log_prob),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="detection"),
        ),
        process_wrapper(
            FuzzingScorer(client, tokenizer=dataset.tokenizer, batch_size=shown_examples,verbose=False,log_prob=log_prob),
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
    parser.add_argument("--shown_examples", type=int, default=5)
    parser.add_argument("--scorer_size", type=str, default="70b")
    parser.add_argument("--model", type=str, default="128k")
    parser.add_argument("--module", type=str, default=".transformer.h.0")
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--start_feature", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--quantization", type=str, default="awq",choices=["awq","bnb","normal"])
    parser.add_argument("--random", type=bool, default=False)
    parser.add_argument("--prob",action="store_false")
    parser.add_arguments(ExperimentConfig, dest="experiment_options")
    parser.add_arguments(FeatureConfig, dest="feature_options")
    args = parser.parse_args()
    experiment_name = args.experiment_name
    


    main(args)
