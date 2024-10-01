import json
import sys
import inspect
from tqdm.auto import tqdm
from collections import Counter
import pandas as pd

from pathlib import Path
from typing import Callable, Literal
import fire
import torch
from ..config import ExperimentConfig, FeatureConfig
from ..features import (
    FeatureDataset,
    FeatureLoader
)
from ..autoencoders.OpenAI.model import TopK, ACTIVATIONS_CLASSES, Autoencoder
from ..features.constructors import default_constructor
from ..features.samplers import sample
from ..autoencoders.DeepMind.model import JumpReLUSAE
from . import (
    ExplainerNeuronFormatter, 
    ExplainerInterventionExample, 
    get_explainer_prompt, 
    fs_examples, 
    garbage_collect, 
    get_git_info, 
    expl_given_generation_score, 
    LAYER_TO_L0
)
from ..features import FeatureDataset
from functools import partial
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from huggingface_hub import hf_hub_download
import numpy as np


PATH_ROOT = Path(__file__).parent.parent.parent


def get_feature_loader(feat_layer, n_feats, n_train, n_test, n_quantiles, width, latents: Literal["sae", "random"] = "sae"):
    module = f".model.layers.{feat_layer}"
    if latents == "sae":
        assert width == 131072
        raw_dir = PATH_ROOT / "cache/gemma_sae_131k"    
    else:
        raw_dir = PATH_ROOT / "cache/gemma_topk"
    feature_dict = {f"{module}": torch.arange(0, n_feats)}
    feature_cfg = FeatureConfig(width=width, n_splits=5, max_examples=100000, min_examples=200)
    experiment_cfg = ExperimentConfig(n_random=0, example_ctx_len=64, n_quantiles=n_quantiles, n_examples_test=0, n_examples_train=(n_train + n_test) // n_quantiles, train_type="quantiles", test_type="even")

    dataset = FeatureDataset(
            raw_dir=str(raw_dir),
            cfg=feature_cfg,
            modules=[module],
            features=feature_dict,  # type: ignore
    )

    constructor=partial(
                default_constructor,
                tokens=dataset.tokens,  # type: ignore
                n_random=experiment_cfg.n_random, 
                ctx_len=experiment_cfg.example_ctx_len, 
                max_examples=feature_cfg.max_examples
            )

    sampler=partial(sample,cfg=experiment_cfg)
    loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)
    return loader


def get_avg_kl(feat_idx, feat_layer, ids_s, intervention_strength, clamp_value, get_subject_logits):
    avg_kl = 0.0
    for ids in ids_s:
        intervened_logps = get_subject_logits(ids, feat_layer, intervention_strength=intervention_strength, clamp_value=clamp_value, feat_idx=feat_idx).log_softmax(dim=0)
        clean_logps = get_subject_logits(ids, feat_layer, intervention_strength=None, clamp_value=None, feat_idx=feat_idx).log_softmax(dim=0)  # no intervention
        kl = (clean_logps.exp() * (clean_logps - intervened_logps)).sum()
        avg_kl += kl

    avg_kl /= len(ids_s)
    return avg_kl


def tune_intervention_strength(
        feat_idx,
        feat_layer,
        ids_s,
        kl_threshold,
        get_subject_logits: Callable,
):
    # sometimes it's not exactly monotonic, but this is usually quite close to kl_threshold because the deltas are small
    
    # do a binary search to get KL of kl_threshold
    min_log_str, max_log_str = -3.0, 4.0
    log_str = 2.0
    rtol = 0.1
    avg_kls = dict()
    for i in range(10):
        intervention_strength = 10 ** log_str
        avg_kl = get_avg_kl(feat_idx, feat_layer, ids_s, intervention_strength, clamp_value=None, get_subject_logits=get_subject_logits)
        avg_kls[intervention_strength] = float(avg_kl)
        if np.isclose(float(avg_kl), kl_threshold, rtol=rtol):
            break
        elif avg_kl > kl_threshold:
            max_log_str = log_str
        else:
            min_log_str = log_str
            
        log_kls = np.log(np.array(sorted(list(avg_kls.values()))))
        log_strs = np.log10(np.array(sorted(list(avg_kls.keys()), key=lambda k: avg_kls[k])))
        if (min(log_kls) <= np.log(kl_threshold) <= max(log_kls)):
            log_str = np.interp([np.log(kl_threshold)], log_kls, log_strs)[0]
        else:
            log_str = (min_log_str + max_log_str) / 2
    else:
        print(f"Binary search for intervention strength did not converge after {i + 1} iterations")
        print(f"Using {intervention_strength} with KL {avg_kl} when target is {kl_threshold}")
        print(f"{avg_kls=}")

    return float(intervention_strength), avg_kls


def consume_all_args(func):
    """Fire doesn't raise an error when it encounters unconsumed arguments, it just ignores them.
    This decorator makes it so that we raise an error when unconsumed arguments are found."""
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        try:
            bound_args = sig.bind(*args, **kwargs)
        except TypeError as e:
            raise ValueError(f"Unconsumed arguments found: {e}")
        
        return func(*bound_args.args, **bound_args.kwargs)
    return wrapper


@consume_all_args
def main(
    device = "cuda",
    n_tokens_per_explainer_example = 5,
    feat_layer = 32,
    n_train=10,
    n_test=40,
    n_quantiles=5,
    n_feats=300,
    kl_threshold = 0.5,
    sae_width = 131072,
    max_generation_length = 8,
    n_explanations = 5,
    explainer_name = "meta-llama/Meta-Llama-3.1-8B",
    run_prefix = "default",
    raise_if_exists = True,
    latents: Literal["sae", "random"] = "sae",
    zero_ablate: bool = False,
):
    assert latents in ["sae", "random"]
    subject_name = "google/gemma-2-9b"
    config = locals().copy()
    config.pop("device")
    config["git_info"] = get_git_info()
    config["run_command"] = ' '.join(sys.argv)

    save_dir = PATH_ROOT / f"counterfactual_results/{run_prefix}_{subject_name.split('/')[-1]}"
    save_path = save_dir / "generations.json"
    config_save_path = save_dir / "config.json"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(config_save_path, "w") as f:
        json.dump(config, f)
    if raise_if_exists:
        assert not (save_dir / "generations_scores.json").exists() or "debug" in run_prefix, f"Save path {save_path} already exists"

    loader = get_feature_loader(feat_layer, n_feats, n_train, n_test, n_quantiles, sae_width, latents=latents)

    subject = AutoModelForCausalLM.from_pretrained(subject_name, torch_dtype=torch.bfloat16).to(device)
    subject_tokenizer = AutoTokenizer.from_pretrained(subject_name)
    subject_tokenizer.pad_token = subject_tokenizer.eos_token
    subject.config.pad_token_id = subject_tokenizer.eos_token_id
    subject_layers = subject.model.layers
    
    if latents == "sae":
        sae_width_str = "131k" if sae_width == 131072 else None
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-9b-pt-res",
            filename=f"layer_{feat_layer}/width_{sae_width_str}/average_l0_{LAYER_TO_L0[feat_layer]}/params.npz",
            force_download=False,
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).to(device).to(torch.bfloat16) for k, v in params.items()}   
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1]).to(device).to(torch.bfloat16)
        sae.load_state_dict(pt_params)
    
    else:
        k = 50
        params = torch.load(PATH_ROOT / f"cache/gemma_topk/layer_{feat_layer}/width_{sae_width}/k_{k}_seed_42/params.pt")
        sae = Autoencoder(sae_width, params["encoder.weight"].shape[1], activation=ACTIVATIONS_CLASSES["TopK"](k=k), normalize=False, tied=False)
        sae.encoder.weight.data = params["encoder.weight"]
        sae.decoder.weight.data = params["decoder.weight"]
        sae.pre_bias.data = params["pre_bias"]
        sae.latent_bias.data = params["latent_bias"]
        sae.to(device).to(subject.dtype)


    def intervene_on_feat(module, input, output, feat_idx=None, intervention_strength=None, clamp_value=None, position: int | slice = slice(0, 0)):
        if intervention_strength is None and clamp_value is None:
            return
        assert intervention_strength is None or clamp_value is None
        hiddens = output[0]  # the later elements of the tuple are the key value cache
        encoding = sae.encode(hiddens)
        if latents == "random":
            encoding = encoding[0]  # OpenAI's topK SAE also returns "info"
        error = hiddens - sae.decode(encoding)  # type: ignore
        if intervention_strength is not None:
            encoding[:, position, feat_idx] += intervention_strength  # type: ignore
        else:
            encoding[:, position, feat_idx] = clamp_value  # type: ignore
        hiddens = sae.decode(encoding) + error  # type: ignore
        return (hiddens, *output[1:])

    def get_first_activating_text(example):
        first_act_idx = (example.activations > 0).nonzero(as_tuple=True)[0][0]
        max_act = example.activations.max()
        return example.tokens[:first_act_idx + 1], max_act.item()
    
    def get_subject_logits(ids, layer, intervention_strength=None, clamp_value=None, position=-1, feat_idx=None):
        for l in range(len(subject_layers)):
            subject_layers[l]._forward_hooks.clear()
        subject_layers[layer].register_forward_hook(partial(intervene_on_feat, intervention_strength=intervention_strength, clamp_value=clamp_value, position=position, feat_idx=feat_idx))

        with torch.inference_mode():
            outputs = subject(ids.unsqueeze(0).to(device))

        return outputs.logits[0, -1, :]
    
    def generate_with_intervention(ids, layer, intervention_strength=None, clamp_value=None, feat_idx=None):
        for l in range(len(subject_layers)):
            subject_layers[l]._forward_hooks.clear()
        
        inputs = ids.to(device).unsqueeze(0)
        intervention_start = inputs.shape[1] - 1
        subject_layers[layer].register_forward_hook(partial(intervene_on_feat, intervention_strength=intervention_strength, clamp_value=clamp_value, position=slice(intervention_start, None), feat_idx=feat_idx))
        with torch.inference_mode():
            out = subject.generate(
                inputs,
                max_new_tokens=max_generation_length,
                num_return_sequences=1,
                temperature=1.0,
                do_sample=True,
            )

        input = subject_tokenizer.decode(ids)
        return subject_tokenizer.decode(out[0]).removeprefix(subject_tokenizer.bos_token).removeprefix(input)
    
    def load_explainer():
        subject.cpu()
        if "sae" in locals():
            sae.cpu()
        garbage_collect()

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

    if save_path.exists():
        explainer, explainer_tokenizer = load_explainer()
    else:
        # random.sample is without replacement
        pools = [random.sample(record.train, len(record.train)) for record in loader]  # type: ignore
        all_explainer_examples = [[get_first_activating_text(e) for e in pool[:n_train]] for pool in pools]
        all_scorer_examples = [[get_first_activating_text(e) for e in pool[n_train:]] for pool in pools]

        all_results = [dict() for _ in range(len(all_scorer_examples))]

        # get intervention strengths
        # we use the test set to tune the intervention strengths because we want the KL to be ~exactly `kl_threshold` on the test set     
        for iter, (record, scorer_examples) in enumerate(tqdm(zip(loader, all_scorer_examples), desc="Tuning intervention strengths", total=len(all_scorer_examples))):
            garbage_collect()
            feat_idx = record.feature.feature_index
            ids_s = [ids for ids, _ in scorer_examples]
            if zero_ablate:
                intervention_strength = None
                avg_kl = get_avg_kl(feat_idx, feat_layer, ids_s, None, clamp_value=0, get_subject_logits=get_subject_logits)
            else:
                intervention_strength, avg_kls = tune_intervention_strength(feat_idx, feat_layer, ids_s, kl_threshold, get_subject_logits)
                avg_kl = avg_kls[intervention_strength]
            all_results[iter].update({
                "intervention_strength": intervention_strength,
                "avg_kl": float(avg_kl),
            })

        # do generation
        for iter, (record, scorer_examples) in enumerate(tqdm(zip(loader, all_scorer_examples), desc="Generating completions", total=len(all_scorer_examples))):
            garbage_collect()
            feat_idx = record.feature.feature_index
            
            # get completions
            completions = []
            for ids, max_act in scorer_examples:            
                # get generation with and without intervention
                prompt = subject_tokenizer.decode(ids)
                completions.append({"text": prompt.removeprefix(subject_tokenizer.bos_token), "max_act": float(max_act), "completions": dict()})
                for name, strength in [("clean", None), ("intervened", all_results[iter]["intervention_strength"])]:
                    cv = 0 if name == "intervened" and zero_ablate else None
                    completions[-1]["completions"][name] = generate_with_intervention(ids, feat_layer, intervention_strength=strength, clamp_value=cv, feat_idx=feat_idx)
        
            all_results[iter].update({
                "feat_idx": int(feat_idx),
                "scorer_examples": [subject_tokenizer.decode(ids) for ids, _ in scorer_examples],
                "completions": completions,
            })
            
        # get intervention results
        for iter, (record, explainer_examples) in enumerate(tqdm(zip(loader, all_explainer_examples), desc="Running interventions for explainer", total=len(all_explainer_examples))):
            garbage_collect()        
            feat_idx = record.feature.feature_index

            intervention_examples = []
            for ids, act in explainer_examples:
                clean_logits = get_subject_logits(ids, feat_layer, intervention_strength=None, clamp_value=None, feat_idx=feat_idx)
                intervened_logits = get_subject_logits(ids, feat_layer, intervention_strength=all_results[iter]["intervention_strength"], clamp_value=0 if zero_ablate else None, feat_idx=feat_idx)
                top_probs = (intervened_logits.softmax(dim=-1) - clean_logits.softmax(dim=-1)).topk(n_tokens_per_explainer_example)
                
                top_tokens = [subject_tokenizer.decode(i) for i in top_probs.indices]
                top_p_increases = top_probs.values.tolist()
                prompt = subject_tokenizer.decode(ids)
                intervention_examples.append(
                    ExplainerInterventionExample(
                        prompt=prompt.removeprefix(subject_tokenizer.bos_token),
                        top_tokens=top_tokens,
                        top_p_increases=top_p_increases
                    )
                )
            all_results[iter].update({
                "explainer_intervention_examples": intervention_examples,
                "explainer_examples": [subject_tokenizer.decode(ids) for ids, _ in explainer_examples],
            })
            if (iter - 1) % 10 == 0:
                all_df = pd.DataFrame(all_results)
                all_df.to_json(save_path)
        all_df = pd.DataFrame(all_results)
        all_df.to_json(save_path)

        explainer, explainer_tokenizer = load_explainer()

        for iter, row in enumerate(tqdm(all_results, desc="Generating explanations")):
            garbage_collect()
            intervention_examples = row["explainer_intervention_examples"]
            neuron_prompter = ExplainerNeuronFormatter(intervention_examples=intervention_examples)

            explainer_prompt = get_explainer_prompt(neuron_prompter, fs_examples)            
            explainer_input_ids = explainer_tokenizer(explainer_prompt, return_tensors="pt").input_ids.to(device)
            with torch.inference_mode():
                samples = explainer.generate(
                    explainer_input_ids,
                    max_new_tokens=20,
                    eos_token_id=explainer_tokenizer.encode("\n")[-1],
                    num_return_sequences=n_explanations,
                    temperature=0.7,
                    do_sample=True,
                )[:, explainer_input_ids.shape[1]:]
            explanations = [explainer_tokenizer.decode(sample).split("\n")[0].strip() for sample in samples]

            all_results[iter].update({
                "explanations": explanations,
                "explainer_prompts": [example.prompt for example in intervention_examples],
                "neuron_prompter": neuron_prompter,
            })
            if (iter - 1) % 10 == 0:
                all_df = pd.DataFrame(all_results)
                all_df.to_json(save_path)
        all_df = pd.DataFrame(all_results)
        all_df.to_json(save_path)

    # score
    # NOTE: we currently require the scorer is the same as the explainer
    expl_given_generation_score(explainer, explainer_tokenizer, str(save_path), device)
    return str(save_dir)
