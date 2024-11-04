import asyncio
import os
import random
from functools import partial

import orjson
import torch
from simple_parsing import ArgumentParser
from transformers import  AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch 

from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import explanation_loader, random_explanation_loader
from sae_auto_interp.features import (
    FeatureDataset,
    FeatureLoader
)
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import EmbeddingScorer


def main(args):
    module = args.module
    feature_cfg = args.feature_options
    experiment_cfg = args.experiment_options
    shown_examples = args.shown_examples
    n_features = args.features  
    start_feature = args.start_feature
    sae_model = args.model
    random = args.random
    experiment_name = args.experiment_name
    ### Load dataset ###
    feature_dict = {f"{module}": torch.arange(start_feature,start_feature+n_features)}
    dataset = FeatureDataset(
        raw_dir=f"raw_features/{sae_model}",
        cfg=feature_cfg,
        modules=[module],
        features=feature_dict,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
    
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
    if random:
        explainer = partial(random_explanation_loader, explanation_dir=f"results/explanations/{sae_model}/{experiment_name}/")
        results_dir = f"results/scores/{sae_model}/random_explanation/"

    else:
        explainer = partial(explanation_loader, explanation_dir=f"results/explanations/{sae_model}/{experiment_name}/")
        results_dir = f"results/scores/{sae_model}/{experiment_name}/"
        
    ### Build Scorer pipe ###

    def scorer_preprocess(result):
        record = result.record
        
        record.explanation = result.explanation
        record.extra_examples = record.random_examples


        return record
    if random:
        experiment_name = "random-explanation"
    def scorer_postprocess(result, score_dir):
        with open(f"{results_dir}/{score_dir}/{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))
    print(results_dir)
    os.makedirs(f"{results_dir}/embedding", exist_ok=True)
    
    model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
    #model = SentenceTransformer("nvidia/NV-Embed-v2", trust_remote_code=True).cuda()
    scorer_pipe = Pipe(
        process_wrapper(
            EmbeddingScorer(model,tokenizer, batch_size=shown_examples,verbose=False),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="embedding"),
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
    parser.add_arguments(ExperimentConfig, dest="experiment_options")
    parser.add_arguments(FeatureConfig, dest="feature_options")
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()
    


    main(args)
