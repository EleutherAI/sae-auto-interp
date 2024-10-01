import random
from collections import Counter
import warnings
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
from pathlib import Path

from sae_auto_interp.counterfactuals.pipeline import expl_given_generation_score

def load_explainer(explainer_name, device):
    # get explainer results
    explainer = AutoModelForCausalLM.from_pretrained(
        explainer_name,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        )
    explainer_tokenizer = AutoTokenizer.from_pretrained(explainer_name)
    explainer_tokenizer.pad_token = explainer_tokenizer.eos_token
    explainer.config.pad_token_id = explainer_tokenizer.eos_token_id
    explainer.generation_config.pad_token_id = explainer_tokenizer.eos_token_id

    return explainer, explainer_tokenizer


def main(expl_path: Path | str, explainer_name: str = "meta-llama/Meta-Llama-3.1-8B", device: str = "cuda"):
    save_path = Path(str(expl_path).replace("re=False", "re=True"))
    random_explanations_df = pd.read_json(expl_path)
    all_expls = [e for l in random_explanations_df["explanations"].tolist() for e in l]
    if Counter(all_expls).most_common(1)[0][1] / len(all_expls) > 0.25:
        warnings.warn("More than 25% of the explanations are the same. Shuffling may not cause enough change in the explanations.")

    # shuffle the explanations column
    random_explanations_df["explanations"] = random.sample(random_explanations_df["explanations"].tolist(), k=len(random_explanations_df))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    random_explanations_df.to_json(save_path, orient="records")

    explainer, explainer_tokenizer = load_explainer(explainer_name, device)

    expl_given_generation_score(explainer, explainer_tokenizer, str(save_path), device)
    return str(save_path.parent)
