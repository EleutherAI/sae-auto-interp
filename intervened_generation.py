from tqdm.auto import tqdm
from itertools import product, islice
import time
from collections import Counter
import pandas as pd

import fire
import torch
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.features import (
    FeatureDataset,
    FeatureLoader
)
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.counterfactuals import ExplainerNeuronFormatter, ExplainerInterventionExample, get_explainer_prompt, few_shot_prompts, few_shot_explanations, few_shot_generations, scorer_separator, JumpReLUSAE, fs_examples
from sae_auto_interp.features import FeatureDataset
from functools import partial
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from huggingface_hub import hf_hub_download
import numpy as np



def garbage_collect():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_feature_loader(feat_layer, n_feats, sae_model, n_train, n_test, n_quantiles):
    module = f".model.layers.{feat_layer}"
    feature_dict = {f"{module}": torch.arange(0, n_feats)}
    feature_cfg = FeatureConfig(width=131072, n_splits=5, max_examples=100000, min_examples=200)
    experiment_cfg = ExperimentConfig(n_random=0, example_ctx_len=64, n_quantiles=n_quantiles, n_examples_test=0, n_examples_train=(n_train + n_test) // n_quantiles, train_type="quantiles", test_type="even")

    dataset = FeatureDataset(
            raw_dir=f"/mnt/ssd-1/gpaulo/SAE-Zoology/raw_features/{sae_model}",
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

def main(
    device = "cuda:6",
    n_tokens_per_explainer_example = 5,
    explainer_intervention_strength = 32,
    feat_layer = 32,
    sae_model = "gemma/131k",
    n_train=5,
    n_test=10,
    n_quantiles=5,
    n_feats=10,
    max_generation_length = 8,
    n_explanations = 1,
    steering_strength = 10,
    random_resid_direction = False,
    random_explanations = False,
    explainer_name = "meta-llama/Meta-Llama-3.1-8B",
):
    loader = get_feature_loader(feat_layer, n_feats, sae_model, n_train, n_test, n_quantiles)

    subject_name = "google/gemma-2-9b"
    subject = AutoModelForCausalLM.from_pretrained(subject_name).to(device)
    subject_tokenizer = AutoTokenizer.from_pretrained(subject_name)
    subject_tokenizer.pad_token = subject_tokenizer.eos_token
    subject.config.pad_token_id = subject_tokenizer.eos_token_id
    subject_layers = subject.model.layers
    
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-9b-pt-res",
        filename=f"layer_{feat_layer}/width_131k/average_l0_51/params.npz",
        force_download=False,
    )

    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}

    def get_decoder_weight(feat_idx, device, random_resid_direction):
        decoder_feat = pt_params["W_dec"][feat_idx, :]
        if random_resid_direction:
            return torch.randn_like(decoder_feat)
        return decoder_feat
        
    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1]).to(device)
    sae.load_state_dict(pt_params)

    def addition_intervention(module, input, output, intervention_strength=10.0, position: int | slice = -1, feat=None):
        hiddens = output[0]  # the later elements of the tuple are the key value cache
        hiddens[:, position, :] += intervention_strength * feat.to(hiddens.device)  # type: ignore

    def clamping_intervention(module, input, output, feat_idx=None, clamp_value=0.0, position: int | slice = slice(None)):
        hiddens = output[0]  # the later elements of the tuple are the key value cache
        
        encoding = sae.encode(hiddens)
        error = hiddens - sae.decode(encoding)
        encoding[:, position, feat_idx] = clamp_value
        hiddens = sae.decode(encoding) + error
        return (hiddens, *output[1:])

    def get_preactivating_text(example):
        first_act_idx = (example.activations > 0).nonzero(as_tuple=True)[0][0]
        idx = random.randint(max(0, first_act_idx - 1 - max_generation_length // 2), first_act_idx - 1) if first_act_idx > 0 else 0
        max_act = example.activations.max()
        return subject_tokenizer.decode(example.tokens[:idx]), max_act.item()
    
    def get_subject_logits(text, layer, intervention_strength=0.0, position=-1, feat=None):
        for l in range(len(subject_layers)):
            subject_layers[l]._forward_hooks.clear()
        subject_layers[layer].register_forward_hook(partial(addition_intervention, intervention_strength=intervention_strength, position=-1, feat=feat))

        inputs = subject_tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.inference_mode():
            outputs = subject(**inputs)

        return outputs.logits[0, -1, :]
    
    def generate_with_intervention(text, layer, clamp_value=0.0, feat_idx=None):
        for l in range(len(subject_layers)):
            subject_layers[l]._forward_hooks.clear()
        
        inputs = subject_tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        # x[:, slice(None), :] is equivalent to x[:, :, :]
        subject_layers[layer].register_forward_hook(partial(clamping_intervention, clamp_value=clamp_value, feat_idx=feat_idx))
        with torch.inference_mode():
            out = subject.generate(
                **inputs,
                max_new_tokens=max_generation_length,
                num_return_sequences=1,  # TODO: maybe things could speed up if we use larger n, or batch sizes
                temperature=1.0,
                do_sample=True,
            )

        return subject_tokenizer.decode(out[0]).removeprefix(subject_tokenizer.bos_token).removeprefix(text)
    
    
    random_explanations_source = "counterfactual_results/generative_gemma-2-9b_32layer_150feats.json"
    random_explanations_df = pd.read_json(random_explanations_source)
    random_explanations_pool = [e for expl in random_explanations_df["explanations"] for e in expl]
    random.shuffle(random_explanations_pool)

    pools = [random.sample(record.train, len(record.train)) for record in loader]  # type: ignore
    all_explainer_examples = [[get_preactivating_text(e) for e in pool[:n_train]] for pool in pools]
    all_scorer_examples = [[get_preactivating_text(e) for e in pool[n_train:]] for pool in pools]

    modifiers = ('_random_dir' if random_resid_direction else '') + ('_random_expl' if random_explanations else '')
    save_path = f"counterfactual_results/generative_{subject_name.split('/')[-1]}_{feat_layer}layer_{n_feats}feats{modifiers}.json"
    all_results = []
    
    # get intervention results
    for iter, (record, explainer_examples) in enumerate(tqdm(zip(loader, all_explainer_examples), desc="Intervention")):
        garbage_collect()        
        feat_idx = record.feature.feature_index
        decoder_feat = get_decoder_weight(feat_idx, device, random_resid_direction)

        intervention_examples = []
        for prompt, act in explainer_examples:
            clean_logits = get_subject_logits(prompt, feat_layer, intervention_strength=0.0, feat=decoder_feat)
            intervened_logits = get_subject_logits(prompt, feat_layer, intervention_strength=explainer_intervention_strength, feat=decoder_feat)
            top_probs = (intervened_logits.softmax(dim=-1) - clean_logits.softmax(dim=-1)).topk(n_tokens_per_explainer_example)
            
            top_tokens = [subject_tokenizer.decode(i) for i in top_probs.indices]
            top_p_increases = top_probs.values.tolist()
            intervention_examples.append(
                ExplainerInterventionExample(
                    prompt=prompt,
                    top_tokens=top_tokens,
                    top_p_increases=top_p_increases
                )
            )
        all_results.append({
            "feat_idx": feat_idx,
            "explainer_intervention_examples": intervention_examples,
        })

    subject.cpu()
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

    for iter, row in enumerate(tqdm(all_results)):
        garbage_collect()
        intervention_examples = row["explainer_intervention_examples"]
        neuron_prompter = ExplainerNeuronFormatter(intervention_examples=intervention_examples)

        # TODO: improve the few-shot examples
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
        explanations = Counter([explainer_tokenizer.decode(sample).split("\n")[0].strip() for sample in samples])

        for ie in intervention_examples:
            print(ie.top_tokens)
            print(ie.top_p_increases)
        print(explanations)

        all_results[iter].update({
            "explanations": dict(explanations),
            "explainer_prompts": [example.prompt for example in intervention_examples],
            "explainer_examples": explainer_examples,
            "neuron_prompter": neuron_prompter,
        })

    explainer.cpu()
    garbage_collect()
    subject.to(device)
    sae.to(device)

    # do generation
    for iter, (row, scorer_examples) in enumerate(tqdm(zip(all_results, all_scorer_examples), desc="Generation")):
        garbage_collect()
        generation_time = time.time()
        # get completions
        completions = []
        for prompt, max_act in scorer_examples:            
            # get generation with and without intervention
            completions.append({"text": prompt, "max_act": max_act, "completions": dict()})
            for name, strength in [("clean", 0), ("intervened", steering_strength * max_act)]:
                completions[-1]["completions"][name] = generate_with_intervention(prompt, feat_layer, clamp_value=strength, feat_idx=feat_idx)
        print(completions)
        print(f"Generation took {time.time() - generation_time:.2f} seconds")

        all_results[iter].update({
            "scorer_examples": scorer_examples,
            "completions": completions,
        })
        if (iter - 1) % 10 == 0:
            all_df = pd.DataFrame(all_results)
            all_df.to_json(save_path)
    all_df = pd.DataFrame(all_results)
    all_df.to_json(save_path)


# def explain():
#     # TODO
    
#     # scorer_name = "google/gemma-2-27b"
#     scorer_name = "meta-llama/Meta-Llama-3.1-8B"
#     scorer = AutoModelForCausalLM.from_pretrained(
#         scorer_name,
#         device_map={"": device},
#         torch_dtype=torch.bfloat16,
#         quantization_config=BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#         )
#     )
#     scorer_tokenizer = AutoTokenizer.from_pretrained(scorer_name)
#     scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
#     scorer.config.pad_token_id = scorer_tokenizer.eos_token_id
#     scorer.generation_config.pad_token_id = scorer_tokenizer.eos_token_id


# def score():
#     # TODO
#     # get KV cache for the scoring few-shot prompt
#     scorer_prompt = get_scorer_surprisal_prompt(few_shot_prompts[0], few_shot_generations[0], few_shot_explanations[0], few_shot_prompts, few_shot_explanations, few_shot_generations)
#     scorer_fs_prefix = scorer_prompt[:scorer_prompt.rfind(scorer_separator)]
#     scorer_fs_ids = scorer_tokenizer(scorer_fs_prefix, return_tensors="pt").input_ids.to(device)

#     with torch.inference_mode():
#         out = scorer(scorer_fs_ids, return_dict=True, use_cache=True)
#     scorer_fs_kv = out.past_key_values


if __name__ == "__main__":
    fire.Fire(main)