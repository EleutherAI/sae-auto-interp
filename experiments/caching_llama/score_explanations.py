import asyncio
import os
from functools import partial

import orjson
import torch
from simple_parsing import ArgumentParser

from sae_auto_interp.clients import Local
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import explanation_loader
from sae_auto_interp.features import (
    FeatureDataset,
)
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import DetectionScorer, FuzzingScorer
from sae_auto_interp.utils import (
    load_tokenized_data,
    load_tokenizer,
)


def main(args):
    module = args.module
    feature_cfg = args.feature_options
    experiment_cfg = args.experiment_options
    shown_examples = args.shown_examples
    n_features = args.features  
    start_feature = args.start_feature
    sae_model = args.model
    quantization = args.quantization

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

    ### Load client ###
    
    if quantization == "awq":
        client = Local("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",base_url=f"http://localhost:{args.port}/v1")
    else:
        client = Local("meta-llama/Meta-Llama-3.1-70B-Instruct",base_url=f"http://localhost:{args.port}/v1")
 

    EXPLAINER_OUT_DIR = f"results/explanations/{sae_model}_{experiment_name}"

    ### Build Explainer pipe ###
    explainer_pipe = partial(explanation_loader, explanation_dir=EXPLAINER_OUT_DIR)


    ### Build Scorer pipe ###

    def scorer_preprocess(result):
        record = result.record
        
        record.explanation = result.explanation
        record.extra_examples = record.random_examples

        return record

    def scorer_postprocess(result, score_dir):
        with open(f"results/scores/{sae_model}_{experiment_name}_{score_dir}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    os.makedirs(f"results/scores/{sae_model}_{experiment_name}_detection", exist_ok=True)
    os.makedirs(f"results/scores/{sae_model}_{experiment_name}_fuzz", exist_ok=True)

    scorer_pipe = Pipe(
        process_wrapper(
            DetectionScorer(client, tokenizer=tokenizer, batch_size=shown_examples,verbose=False,log_prob=True),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="detection"),
        ),
        process_wrapper(
            FuzzingScorer(client, tokenizer=tokenizer, batch_size=shown_examples,verbose=False,log_prob=True),
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

    asyncio.run(pipeline.run(max_processes=10))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--shown_examples", type=int, default=5)
    parser.add_argument("--model", type=str, default="128k")
    parser.add_argument("--module", type=str, default=".transformer.h.0")
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--start_feature", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--quantization", type=str, default="awq",choices=["awq","bnb","normal"])
    parser.add_arguments(ExperimentConfig, dest="experiment_options")
    parser.add_arguments(FeatureConfig, dest="feature_options")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    
    experiment_name = args.experiment_name
    


    main(args)
