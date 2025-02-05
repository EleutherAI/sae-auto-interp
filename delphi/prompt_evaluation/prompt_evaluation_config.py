from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class PromptConfig:
    name: str
    explainer_prompts: dict[str, str]
    detection_prompts: dict[str, str]
    fuzzing_prompts: dict[str, str]
    simulation_prompts: dict[str, str]
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    @classmethod
    def from_json(cls, path: Path) -> "PromptConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


@dataclass
class PromptEvaluationConfig:
    n_features: int = 100
    batch_size: int = 8
    device: str = "cuda"
    save_dir: Path = Path("results/prompt_experiments")
    

@dataclass 
class PromptEvaluationResult:
    config_name: str
    detection_scores: dict[str, float]
    fuzzing_scores: dict[str, float]
    simulation_scores: dict[str, float]

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
