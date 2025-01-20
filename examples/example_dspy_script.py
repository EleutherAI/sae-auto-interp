import asyncio
import json
import os
import time
from functools import partial
from dataclasses import dataclass, field
from typing import List, Literal, Union, Optional
from sae_auto_interp.dspy_pipeline import (
    train_classifier_pipeline,
    evaluate_classifier_pipeline,
    DSPyClassifierPipeline
)
from pathlib import Path
import dspy

import orjson
import torch
from simple_parsing import ArgumentParser

from sae_auto_interp.clients import Offline
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer
from sae_auto_interp.scorers.classifier.dspy_classifier import (
    DSPyClassifier,
)
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import DetectionScorer, FuzzingScorer
from sae_auto_interp.logger import logger

import asyncio
import logging
import os
import time
from functools import partial

import dotenv
import nest_asyncio
import orjson
import torch
from dspy import LM
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sae_auto_interp.clients import DSPy
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer, DSPyExplainer
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.scorers import DetectionScorer, DSPyClassifier, FuzzingScorer



@dataclass
class DSPyModelConfig:
    optimizer: Literal["none", "mipro", "bootstrap"] = "bootstrap"
    preload_few_shot: bool = False
    use_cot: bool = False
    n_aux_examples: int = 0
    batch_size: int = 5


@dataclass
class DSPyExperimentConfig:
    feature_dir: str
    module: str
    save_dir: Optional[str] = None
    load_dirs: List[str] = field(default_factory=list)

    classification_method: Literal["fuzz", "detect", "pseudo_fuzz"] = "pseudo_fuzz"
    features_train: int = 50
    features_test: int = 250
    
    model_config: DSPyModelConfig = DSPyModelConfig()
    
    lm_provider: Literal["vllm", "openrouter"] = "vllm"
    experiment_options: ExperimentConfig = ExperimentConfig()
    feature_config: FeatureConfig = FeatureConfig()
    
    def save_json(self):
        return orjson.dumps(self.__dict__).decode("utf-8")

    def load_json(s):
        param_dict = orjson.loads(s)
        param_dict["model_config"]  = DSPyModelConfig(**param_dict["model_config"])
        param_dict["experiment_options"] = ExperimentConfig(**param_dict["experiment_options"])
        param_dict["feature_config"] = FeatureConfig(**param_dict["feature_config"])
        return DSPyExperimentConfig(**param_dict)


def dspy_classifier_pipeline_from_config(config: DSPyExperimentConfig):
    return DSPyClassifierPipeline(
        explainer_few_shot=config.model_config.preload_few_shot,
        classifier_few_shot=config.model_config.preload_few_shot,
        n_aux_examples=config.model_config.n_aux_examples,
        batch_size=config.model_config.batch_size,
        ignore_errors=True,
    )


class DSPyExperiment:
    def __init__(self, config: DSPyExperimentConfig):
        self.config = config
        self._load_client()
        self._load_data()
    
    @property
    def tokenizer(self):
        return self._dataset_train.tokenizer
    
    def train(self):
        if self.config.model_config.optimizer == "none":
            module = dspy_classifier_pipeline_from_config(config)
        else:
            module = train_classifier_pipeline(
                self._loader_train,
                self._dataset_train.tokenizer,
                self._lm,
                explainer_few_shot=self.config.model_config.preload_few_shot,
                classifier_few_shot=self.config.model_config.preload_few_shot,
                optimizer_method=self.config.model_config.optimizer,
                n_aux_examples=self.config.model_config.n_aux_examples,
                method=self.config.classification_method,
                batch_size=self.config.model_config.batch_size,
                ignore_errors=True,
            )
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        module.save(save_dir / "module", save_program=True)
        with open(save_dir / "config.json", "w") as f:
            f.write(self.config.save_json())
            
    async def evaluate(self):
        # all default settings
        default_dspy_config = DSPyExperimentConfig()
        default_dspy_pipeline = dspy_classifier_pipeline_from_config(default_dspy_config)
        # non-DSPy pipeline
        default_explainer = DefaultExplainer(
            self._client,
            tokenizer=self.tokenizer,
            threshold=0.2,
            activations=True,
            cot=False,
        )
        if self.config.classification_method == "detect":
            default_scorer = DetectionScorer(
                self._client,
                tokenizer=self.tokenizer,
                verbose=True,
                log_prob=False,
                batch_size=5,
            )
        elif self.config.classification_method == "pseudo_fuzz":
            default_scorer = FuzzingScorer(
                self._client,
                tokenizer=self.tokenizer,
                verbose=True,
                log_prob=False,
                batch_size=5,
            )
        else:
            raise ValueError("Only detect and pseudo_fuzz are supported for non-DSPy pipelines")
        def explainer_preprocess(x):
            return x
        def explainer_postprocess(result):
            return result

        def scorer_preprocess(result):
            record = result.record
            record.explanation = result.explanation
            record.extra_examples = record.random_examples
            return record
        def scorer_postprocess(x):
            corrects = [c.correct for c in x.score]
            return list(map(int, corrects))

        async def evaluate_default_pipeline(explainer, scorer):
            explainer_pipe = process_wrapper(
                explainer,
                preprocess=explainer_preprocess,
                postprocess=explainer_postprocess,
            )
            scorer_pipe = process_wrapper(
                scorer,
                preprocess=scorer_preprocess,
                postprocess=scorer_postprocess,
            )
            pipe = Pipeline(
                self._loader_eval,
                explainer_pipe,
                scorer_pipe
            )
            start_time = time.time()
            corrects = await pipe.run(16)
            logger.debug(f"Elapsed time: {time.time() - start_time}")
            logger.debug(f"Accuracy: {sum(map(sum, corrects)) / sum(map(len, corrects))}")
            return [sum(c) / len(c) for c in corrects]

        async def evaluate_dspy_module(module, config):
            explainer = DSPyExplainer(
                self._lm, self.tokenizer, config.model_config.use_cot
            )
            if self.config.classification_method == "detect":
                base_scorer = DetectionScorer(
                    self._client,
                    tokenizer=self.tokenizer,
                    verbose=True,
                    log_prob=False,
                    batch_size=config.model_config.batch_size,
                )
            elif self.config.classification_method == "pseudo_fuzz":
                base_scorer = FuzzingScorer(
                    self._client,
                    tokenizer=self.tokenizer,
                    verbose=True,
                    log_prob=False,
                    batch_size=config.model_config.batch_size,
                    threshold=0.2
                )
            else:
                raise ValueError(
                    "Only detect and pseudo_fuzz are supported for non-DSPy evaluation"
                )
            scorer = DSPyClassifier(
                base_scorer, module=module.classifier, batch_size=config.model_config.batch_size,
                cot=config.model_config.use_cot, few_shot=config.model_config.preload_few_shot,
                n_aux_examples=config.model_config.n_aux_examples
            )
            accuracies_pipeline = await evaluate_default_pipeline(explainer, scorer)
            return accuracies_pipeline

        # DSPy modules
        for load_dir in self.config.load_dirs:
            with open(Path(load_dir) / "config.json") as f:
                config = DSPyExperimentConfig.load_json(f.read())
            module = dspy.load(Path(load_dir) / "module")
            accuracies_pipeline = await evaluate_dspy_module(module, config)
            print(accuracies_pipeline)
        # default
        accuracies_default = evaluate_default_pipeline(
            default_explainer, default_scorer
        )
        print(accuracies_default)
        # mixtures

    def _load_client(self):
        dotenv.load_dotenv()
        environ = os.environ
        lm_provider = config.lm_provider
        if lm_provider == "openrouter":
            or_model = "meta-llama/llama-3.3-70b-instruct"
            dspy_lm = LM(
                "openrouter/" + or_model,
                api_key=environ["OPENROUTER_API_KEY"],
                num_retries=16,
                # api_base="https://openrouter.ai/v1/",
            )
            client = DSPy(dspy_lm)
        elif lm_provider == "vllm":
            dspy_lm = LM(
                "openai/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
                api_base="http://localhost:8000/v1/",
                api_key="placeholder",
                # cache=False,
            )
            client = DSPy(dspy_lm)
        elif lm_provider == "together":
            dspy_lm = LM(
                "openai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                api_key=environ["TOGETHER_API_KEY"],
                api_base="https://api.together.xyz/v1/",
            )
            client = DSPy(dspy_lm)
        elif lm_provider == "groq":
            dspy_lm = LM(
                "llama-3.3-70b-specdec",
                api_key=environ["GROQ_API_KEY"],
                api_base="https://api.groq.com/openai/v1/",
            )
            client = DSPy(dspy_lm)
        self._lm = dspy_lm
        self._client = client

    def _load_data(self):
        config = self.config
        feature_cfg = config.feature_config
        experiment_cfg = config.experiment_options
        module = config.module
        feature_dict_train = {f"{module}": torch.arange(config.features_train)}
        feature_dict_eval = {
            f"{module}": torch.arange(
                config.features_train, config.features_train + config.features_test
            )
        }
        dataset_train = FeatureDataset(
            raw_dir=config.feature_dir,
            cfg=feature_cfg,
            modules=[module],
            features=feature_dict_train,
        )
        dataset_eval = FeatureDataset(
            raw_dir=config.feature_dir,
            cfg=feature_cfg,
            modules=[module],
            features=feature_dict_eval,
        )
        self._dataset_train = dataset_train
        self._dataset_eval = dataset_eval

        constructor = partial(
            default_constructor,
            token_loader=dataset_train.load_tokens,  # doesn't matter which dataset we use
            n_random=experiment_cfg.n_random,
            ctx_len=experiment_cfg.example_ctx_len,
            max_examples=feature_cfg.max_examples,
        )
        sampler = partial(sample, cfg=experiment_cfg)
        loader_train = FeatureLoader(
            dataset_train, constructor=constructor, sampler=sampler
        )
        loader_eval = FeatureLoader(
            dataset_eval, constructor=constructor, sampler=sampler
        )
        self._loader_train = loader_train
        self._loader_eval = loader_eval
        
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default="raw_features/new")
    parser.add_argument("--model", type=str, default="gemma/16k")
    parser.add_argument("--module", type=str, default=".model.layers.10")
    parser.add_arguments(ExperimentConfig, dest="experiment_options", default=ExperimentConfig(
        train_type="quantiles", test_type="quantiles", n_examples_train=25, n_examples_test=25, n_quantiles=5,
    ))
    parser.add_arguments(FeatureConfig, dest="feature_options")
    parser.add_argument(
        "--classification_method",
        type=str,
        choices=["fuzz", "detect", "pseudo_fuzz"],
        default="pseudo_fuzz",
        help="Method for corrupting the data for classification",
    )
    parser.add_argument("--features_train", type=int, default=20, help="Number of features in the training set")
    parser.add_argument("--features_test", type=int, default=100, help="Number of features in the test set")
    parser.add_argument("--lm_provider", type=str, choices=["vllm", "openrouter", "together", "groq"], default="vllm")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--n_aux_examples",
        type=int,
        default=10,
        help="Number of auxiliary train examples to show the classifier",
    )
    train_parser.add_argument(
        "--preload_few_shot",
        action="store_true",
        help="Whether to use hand-written preloaded few-shot examples for the explainer and classifier",
    )
    train_parser.add_argument(
        "--use_cot",
        action="store_true",
        help="Whether to use Chain of Thought for the explainer and classifier",
    )
    train_parser.add_argument(
        "--optimizer",
        type=str,
        choices=["none", "mipro", "bootstrap"],
        default="bootstrap",
        help="DSPy optimizer to use",
    )
    train_parser.add_argument(
        "--save_dir",
        type=str,
        help="Where to store results after training",
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for classifier"
    )

    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", aliases=["evaluate"])
    eval_parser.add_argument(
        "--load_dirs",
        nargs="+",
        help="Directories to load completed experiments from for evaluation",
    )
    eval_parser.add_argument(
        "--save_dir",
        type=str,
        help="Where to store results after evaluation",
    )

    args = parser.parse_args()

    config = DSPyExperimentConfig(
        features_train=args.features_train,
        features_test=args.features_test,
        feature_dir=args.feature_dir,
        module=args.module,
        experiment_options=args.experiment_options,
        feature_config=args.feature_options,
        classification_method=args.classification_method,
        lm_provider=args.lm_provider,
    )
    train = args.command == "train"
    evaluate = args.command in ("eval", "evaluate")
    if train:
        config.model_config = DSPyModelConfig(
            optimizer=args.optimizer,
            preload_few_shot=args.preload_few_shot,
            n_aux_examples=args.n_aux_examples,
            use_cot=args.use_cot,
            batch_size=args.batch_size,
        )
        config.save_dir = args.save_dir
    elif evaluate:
        config.load_dirs = args.load_dirs
    experiment = DSPyExperiment(config)
    
    if train:
        experiment.train()
    elif evaluate:
        asyncio.run(experiment.evaluate())
    else:
        raise ValueError("Unknown command. Use --help for help")
