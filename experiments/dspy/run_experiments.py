import asyncio
import os
os.environ["DSPY_CACHEDIR"] = "/mnt/ssd-1/nev/dspy_cache"
from dataclasses import replace

from .experiment_dspy import (
    DSPyExperiment,
    DSPyExperimentConfig,
    DSPyModelConfig,
    ExperimentConfig,
)

if __name__ == "__main__":
    # os.environ["DSPY_CACHEDIR"] = "~/.cache/dspy"
    # experiments we need
    # no tuning
    # batch size: 1, (10)
    # explainer dropout: (0.0), 1.0
    # n_aux_examples: 0, (10)
    # cot: (false), true
    base_experiment = DSPyExperimentConfig(
        feature_dir="raw_features/new",
        module=".model.layers.10",
        experiment_name="base",
        save_dir="base",
        classification_method="pseudo_fuzz",
        features_train=50,
        features_test=500,
        model_config=DSPyModelConfig(
            optimizer="bootstrap",
            n_aux_examples=0,
            drop_out_explainer_prob=0.0,
            batch_size=5,
            use_cot=False
        ),
        lm_provider="vllm",
        lm_name="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        experiment_options=ExperimentConfig(
            n_examples_train=25, n_examples_test=25, n_quantiles=5,
            train_type="quantiles", test_type="quantiles"
        ),
    )
    ds_config = dict(
        lm_name="casperhansen/deepseek-r1-distill-llama-70b-awq",
        lm_api_address="http://localhost:8001/v1"
    )
    configs = [
        base_experiment,
        replace(
            base_experiment,
            experiment_name="n_aux_10",
            save_dir="n_aux_10",
            model_config=replace(base_experiment.model_config, n_aux_examples=10)
        ),
        replace(
            base_experiment,
            experiment_name="batch_size_1",
            save_dir="batch_size_1",
            model_config=replace(base_experiment.model_config, batch_size=1)
        ),
        replace(
            base_experiment,
            experiment_name="batch_size_1_aux",
            save_dir="batch_size_1_aux",
            model_config=replace(base_experiment.model_config, batch_size=1, n_aux_examples=10)
        ),
        replace(
            base_experiment,
            experiment_name="dropout_1.0",
            save_dir="dropout_1.0",
            model_config=replace(base_experiment.model_config, drop_out_explainer_prob=1.0)
        ),
        replace(
            base_experiment,
            experiment_name="cot",
            save_dir="cot",
            model_config=replace(base_experiment.model_config, use_cot=True)
        ),
        replace(
            base_experiment,
            experiment_name="cot_bs1",
            save_dir="cot_bs1",
            model_config=replace(base_experiment.model_config, use_cot=True, batch_size=1, n_aux_examples=3)
        ),
        # replace(
        #     base_experiment,
        #     experiment_name="deepseek",
        #     save_dir="deepseek",
        #     **ds_config,
        # ),
        replace(
            base_experiment,
            experiment_name="mipro",
            save_dir="mipro",
            model_config=replace(base_experiment.model_config, optimizer="mipro"),
        ),
        replace(
            base_experiment,
            experiment_name="mipro_cot",
            save_dir="mipro_cot",
            model_config=replace(base_experiment.model_config, optimizer="mipro", use_cot=True),
        ),
        replace(
            base_experiment,
            experiment_name="no_optim",
            save_dir="no_optim",
            model_config=replace(base_experiment.model_config, optimizer="none"),
        ),
    ]
    configs = [
        replace(config, save_dir=f"results/dspy_experiments/{config.save_dir}") for config in configs
    ]
    print("Training all", len(configs), "experiments")
    for config in configs:
        if os.path.exists(os.path.join(config.save_dir, "module")):
            print("Experiment", config.experiment_name, "already trained, skipping")
            continue
        print("Training experiment", config.experiment_name)
        DSPyExperiment(config).train()
    asyncio.run(DSPyExperiment(replace(
        base_experiment,
        load_dirs=[config.save_dir for config in configs],
        save_dir="results/dspy_scores"
    )).evaluate())