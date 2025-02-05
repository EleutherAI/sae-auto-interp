# Run evaluation with a prompt config
# python -m sae_auto_interp.prompt_evaluation.cli evaluate sae_auto_interp/prompt_config/baseline.json

# Plot results comparing multiple configs
# python -m sae_auto_interp.prompt_evaluation.cli plot results/prompt_experiments --configs baseline v1 v2

from typing import Optional
from pathlib import Path
import typer
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

from pathlib import Path
from delphi.prompt_evaluation.prompt_evaluation_config import (
    PromptConfig,
    PromptEvaluationResult,
    PromptEvaluationConfig,
)
from delphi.pipeline import Pipeline
from delphi.features import FeatureLoader
from delphi.clients import Offline

from delphi.explainers import DefaultExplainer
from delphi.scorers import FuzzingScorer, DetectionScorer, OpenAISimulator

from delphi.features import FeatureRecord, FeatureDataset



class PromptEvaluator:
    def __init__(self, config: PromptEvaluationConfig):
        self.config = config
        self.save_dir = config.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_prompt_cfg(self, prompt_cfg: PromptConfig) -> PromptEvaluationResult:
        # Load feature dataset similar to counterfactuals/pipeline.py
        loader = self._get_feature_loader()

        # Initialize model and client
        client = Offline(prompt_cfg.model, max_memory=0.8, max_model_len=5120)

        # Build pipeline with custom prompts
        pipeline = self._build_pipeline(loader, client, prompt_cfg)

        # Run evaluation
        results = self._run_evaluation(pipeline, prompt_cfg)

        return results

    def _get_feature_loader(self) -> FeatureLoader:
        """Reference counterfactuals/pipeline.py get_feature_loader()"""
        # TODO Load feature config from prompt experiment config
        # TODO Load experiment config from prompt experiment config
        # TODO Load FeatureDataset config from prompt experiment config

        # Create constructor and sampler
        # constructor = partial(
        #     default_constructor,
        #     tokens=dataset.tokens,
        #     n_random=experiment_cfg.n_random,
        #     ctx_len=experiment_cfg.example_ctx_len,
        #     max_examples=feature_cfg.max_examples
        # )
        
        # sampler = partial(sample, cfg=experiment_cfg)
        
        # return FeatureLoader(dataset, constructor=constructor, sampler=sampler)
        
        # TODO: Implement
        pass

    def _build_pipeline(
        self, loader: FeatureLoader, client: Offline, prompt_cfg: PromptConfig
    ) -> Pipeline:
        """Reference counterfactuals/pipeline.py build_pipeline()"""
        # Initialize explainer with custom prompts
        explainer = DefaultExplainer(
            client,
            tokenizer=loader.dataset.tokenizer,
            system_prompt=prompt_cfg.explainer_prompts["system"],
            few_shot_examples=prompt_cfg.explainer_prompts["few_shot"]
        )

        # Create explainer pipe with post-processing
        def explainer_postprocess(result):
            with open(f"{self.save_dir}/{result.record.feature}_explanation.json", "w") as f:
                json.dump(result.explanation, f)
            return result

        explainer_pipe = process_wrapper(
            explainer,
            postprocess=explainer_postprocess
        )

        # Initialize scorers with custom prompts
        detection_scorer = DetectionScorer(
            client, 
            tokenizer=loader.dataset.tokenizer,
            system_prompt=prompt_cfg.detection_prompts["system"],
            few_shot_examples=prompt_cfg.detection_prompts["few_shot"]
        )
        
        fuzzing_scorer = FuzzingScorer(
            client,
            tokenizer=loader.dataset.tokenizer,
            system_prompt=prompt_cfg.fuzzing_prompts["system"],
            few_shot_examples=prompt_cfg.fuzzing_prompts["few_shot"]
        )

        # TODO: Implement
        pass

    def _run_evaluation(
        self, pipeline: Pipeline, prompt_cfg: PromptConfig
    ) -> PromptEvaluationResult:
        """Reference counterfactuals/pipeline.py run_evaluation()"""
        # TODO: Implement
        pass


app = typer.Typer()


@app.command()
def evaluate(
    config_path: Path = typer.Argument(..., help="Path to prompt config JSON"),
    output_dir: Path = typer.Option(
        "results/prompt_evaluation", help="Output directory"
    ),
    n_features: int = typer.Option(100, help="Number of features to evaluate"), # TODO is this needed?
):
    """Run evaluation pipeline with specified prompt configuration"""
    prompt_cfg = PromptConfig.from_json(config_path)
    evaluator = PromptEvaluator(
        PromptEvaluationConfig(n_features=n_features, save_dir=output_dir)
    )

    results = evaluator.evaluate_prompt_cfg(prompt_cfg)
    results.save(output_dir / f"{prompt_cfg.name}_results.json")


@app.command()
def plot(
    results_dir: Path = typer.Option(
        "results/prompt_evaluation", help="Directory containing experiment results"
    ),
    configs: Optional[list[str]] = typer.Option(None, help="Specific configs to plot"),
    output_path: Path = typer.Option("results/prompt_comparisons/prompt_comparison.pdf", help="Output plot path"),
):
    """Generate comparison plots for experiment results"""
    results = []
    for result_file in results_dir.glob("*_results.json"):
        if configs and result_file.stem not in configs:
            continue
        results.append(PromptEvaluationResult.from_json(result_file))

    plot_comparisons(results, output_path)


def plot_comparisons(results: list[PromptEvaluationResult], output_path: Path):
    """Generate plots comparing metrics across configurations"""
    # Create comparison plots using matplotlib/seaborn
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    names = [r.config_name for r in results]

    # Detection scores
    sns.barplot(x=names, y=[r.detection_scores["mean"] for r in results], ax=ax1)
    ax1.set_title("Detection Scores")

    # Fuzzing scores
    sns.barplot(x=names, y=[r.fuzzing_scores["mean"] for r in results], ax=ax2)
    ax2.set_title("Fuzzing Scores")

    # Simulation scores
    sns.barplot(x=names, y=[r.simulation_scores["mean"] for r in results], ax=ax3)
    ax3.set_title("Simulation Scores")

    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    app()
