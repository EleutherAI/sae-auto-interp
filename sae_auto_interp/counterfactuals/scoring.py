import os

from . import get_scorer_surprisal_prompt, scorer_separator, few_shot_prompts, few_shot_explanations, few_shot_generations, garbage_collect

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def expl_given_generation_score(scorer, scorer_tokenizer, completions_path, device):
    all_results = pd.read_json(completions_path).to_dict(orient="records")
    out_path = completions_path.replace(".json", "_scores.json")
    assert not os.path.exists(out_path) or "debug" in out_path, f"Output path {out_path} already exists"
    
    # get KV cache for the scoring few-shot prompt
    scorer_prompt = get_scorer_surprisal_prompt(few_shot_prompts[0], few_shot_generations[0], few_shot_explanations[0], few_shot_prompts, few_shot_explanations, few_shot_generations)
    scorer_fs_prefix = scorer_prompt[:scorer_prompt.rfind(scorer_separator)]
    scorer_fs_ids = scorer_tokenizer(scorer_fs_prefix, return_tensors="pt").input_ids.to(device)

    with torch.inference_mode():
        out = scorer(scorer_fs_ids, return_dict=True, use_cache=True)
    scorer_fs_kv = out.past_key_values

    for iter, record in enumerate(tqdm(all_results, desc="Scoring completions")):
        garbage_collect()
        surprisals_by_explanation = dict()
        
        delta_conditional_entropy_by_explanation = dict()
        delta_conditional_entropy_sems_by_explanation = dict()
        for explanation in record["explanations"]:
            surprisals = {"zeroed": [], "intervened": []}

            
            for i, (prompt, act) in enumerate(record["scorer_examples"]):
                            
                for name, completion in record["completions"][i]["completions"].items():
                    text, expl_start_idx = get_scorer_surprisal_prompt(
                        record["completions"][i]["text"], completion, explanation, return_explanation_start=True
                    )
                    scorer_prompt, expl = text[:expl_start_idx], text[expl_start_idx:]
                    
                    scorer_prompt_ids = torch.cat([
                        scorer_fs_ids, 
                        scorer_tokenizer(scorer_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                    ], dim=1)
                    scorer_expl_ids = scorer_tokenizer(expl, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                    scorer_input_ids = torch.cat([scorer_prompt_ids, scorer_expl_ids], dim=1)
                    labels = scorer_input_ids.clone()
                    labels[:, :scorer_prompt_ids.shape[1]] = -100
                    with torch.inference_mode():
                        loss = scorer(scorer_input_ids, labels=labels, use_cache=True, past_key_values=scorer_fs_kv).loss
                        # HF averages over the sequence length, so we undo that
                        surprisals[name].append(loss.item() * (scorer_input_ids.shape[1] - scorer_prompt_ids.shape[1]))
                
            surprisals = {k: np.array(v) for k, v in surprisals.items()}
            surprisals_by_explanation[explanation] = surprisals
            delta_conditional_entropy_by_explanation[explanation] = (surprisals["zeroed"] - surprisals["intervened"]).mean()
            delta_conditional_entropy_sems_by_explanation[explanation] = (surprisals["zeroed"] - surprisals["intervened"]).std(ddof=1) / np.sqrt(len(surprisals["intervened"]))

        best_explanation = max(delta_conditional_entropy_by_explanation, key=lambda x: delta_conditional_entropy_by_explanation[x])
        record.update({
            "delta_conditional_entropy_by_explanation": delta_conditional_entropy_by_explanation,
            "delta_conditional_entropy_sems_by_explanation": delta_conditional_entropy_sems_by_explanation,
            "max_delta_conditional_entropy": max(delta_conditional_entropy_by_explanation.values()),
            "best_explanation": best_explanation,
        })
        if iter % 10 == 0:
            pd.DataFrame(all_results).to_json(out_path, orient="records")

    pd.DataFrame(all_results).to_json(out_path, orient="records")
