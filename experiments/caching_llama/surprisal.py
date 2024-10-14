import asyncio
import os
import random
from functools import partial

import orjson
import torch
from simple_parsing import ArgumentParser
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch 

from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import explanation_loader
from sae_auto_interp.features import (
    FeatureDataset,
    FeatureLoader
)
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import SurprisalScorer


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
    feature_dict = {f"{module}": torch.arange(start_feature,start_feature+n_features)}
    dataset = FeatureDataset(
        raw_dir=f"raw_features_{sae_model}",
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
    

    ### Build Explainer pipe ###
  
    explainer = partial(explanation_loader, explanation_dir=f"results/explanations/{sae_model}_{experiment_name}/")

    ### Build Scorer pipe ###

    def scorer_preprocess(result):
        record = result.record
        
        record.explanation = result.explanation
        record.extra_examples = record.random_examples


        return record

    def scorer_postprocess(result, score_dir):
        with open(f"results/scores/{sae_model}_{experiment_name}_{score_dir}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    os.makedirs(f"results/scores/{sae_model}_{experiment_name}_surprisal", exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-70B", device_map="auto",load_in_8bit=True)
    model.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B")
    scorer_pipe = Pipe(
        process_wrapper(
            SurprisalScorer(model,tokenizer, batch_size=shown_examples,verbose=False),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="surprisal"),
        ),
        
    )

    ### Build the pipeline ###

    pipeline = Pipeline(
        loader,
        explainer,
        scorer_pipe,
    )

    asyncio.run(pipeline.run(1))


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
