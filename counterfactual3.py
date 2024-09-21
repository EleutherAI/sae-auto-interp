import torch
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.features import (
    FeatureDataset,
    FeatureLoader
)
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample

feat_layer = 32
sae_model = "gemma/131k"
module = f".model.layers.{feat_layer}"
n_train, n_test, n_quantiles = 5, 160, 5
n_feats = 100
feature_dict = {f"{module}": torch.arange(0, n_feats)}
feature_cfg = FeatureConfig(width=131072, n_splits=5, max_examples=100000, min_examples=200)
experiment_cfg = ExperimentConfig(n_random=0, example_ctx_len=64, n_quantiles=5, n_examples_test=0, n_examples_train=n_train + n_test // n_quantiles, train_type="quantiles", test_type="even")

from sae_auto_interp.features import FeatureDataset
from functools import partial
import random

dataset = FeatureDataset(
        raw_dir=f"/mnt/ssd-1/gpaulo/SAE-Zoology/raw_features/{sae_model}",
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

[i.feature.feature_index for i in loader]
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from huggingface_hub import hf_hub_download
import numpy as np

subject_device = "cuda:6"

subject_name = "google/gemma-2-9b"
subject = AutoModelForCausalLM.from_pretrained(subject_name).to(subject_device)
subject_tokenizer = AutoTokenizer.from_pretrained(subject_name)
subject_tokenizer.pad_token = subject_tokenizer.eos_token
subject.config.pad_token_id = subject_tokenizer.eos_token_id
subject_layers = subject.model.layers
scorer_device = "cuda:7"

# TODO: use llama-3.1-70b instead
scorer_name = "google/gemma-2-27b"
scorer = AutoModelForCausalLM.from_pretrained(
    scorer_name,
    device_map={"": scorer_device},
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
)
scorer_tokenizer = AutoTokenizer.from_pretrained(scorer_name)
scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
scorer.config.pad_token_id = scorer_tokenizer.eos_token_id
scorer.generation_config.pad_token_id = scorer_tokenizer.eos_token_id

# explainer is the same model as the scorer
explainer_device = scorer_device
explainer = scorer
explainer_tokenizer = scorer_tokenizer
from dataclasses import dataclass
import copy

@dataclass
class ExplainerInterventionExample:
    prompt: str
    top_tokens: list[str]
    top_p_increases: list[float]

    def __post_init__(self):
        self.prompt = self.prompt.replace("\n", "\\n")

    def text(self) -> str:
        tokens_str = ", ".join(f"'{tok}' (+{round(p, 3)})" for tok, p in zip(self.top_tokens, self.top_p_increases))
        return f"<PROMPT>{self.prompt}</PROMPT>\nMost increased tokens: {tokens_str}"
    
@dataclass
class ExplainerNeuronFormatter:
    intervention_examples: list[ExplainerInterventionExample]
    explanation: str | None = None

    def text(self) -> str:
        text = "\n\n".join(example.text() for example in self.intervention_examples)
        text += "\n\nExplanation:"
        if self.explanation is not None:
            text += " " + self.explanation
        return text


def get_explainer_prompt(neuron_prompter: ExplainerNeuronFormatter, few_shot_examples: list[ExplainerNeuronFormatter] | None = None) -> str:
    prompt = "We're studying neurons in a transformer model. We want to know how intervening on them affects the model's output.\n\n" \
        "For each neuron, we'll show you a few prompts where we intervened on that neuron at the final token position, and the tokens whose logits increased the most.\n\n" \
        "The tokens are shown in descending order of their probability increase, given in parentheses. Your job is to give a short summary of what outputs the neuron promotes.\n\n"
    
    i = 1
    for few_shot_example in few_shot_examples or []:
        assert few_shot_example.explanation is not None
        prompt += f"Neuron {i}\n" + few_shot_example.text() + "\n\n"
        i += 1

    prompt += f"Neuron {i}\n"
    prompt += neuron_prompter.text()

    return prompt


fs_examples = [
    ExplainerNeuronFormatter(
        intervention_examples=[
            ExplainerInterventionExample(
                prompt="My favorite food is",
                top_tokens=[" oranges", " bananas", " apples"],
                top_p_increases=[0.81, 0.09, 0.02]
            ),
            ExplainerInterventionExample(
                prompt="Whenever I would see",
                top_tokens=[" fruit", " a", " apples", " red"],
                top_p_increases=[0.09, 0.06, 0.06, 0.05]
            ),
            ExplainerInterventionExample(
                prompt="I like to eat",
                top_tokens=[" fro", " fruit", " oranges", " bananas", " strawberries"],
                top_p_increases=[0.14, 0.13, 0.11, 0.10, 0.03]
            )
        ],
        explanation="fruits"
    ),
    ExplainerNeuronFormatter(
        intervention_examples=[
            ExplainerInterventionExample(
                prompt="Once",
                top_tokens=[" upon", " in", " a", " long"],
                top_p_increases=[0.22, 0.2, 0.05, 0.04]
            ),
            ExplainerInterventionExample(
                prompt="Ryan Quarles\\n\\nRyan Francis Quarles (born October 20, 1983)",
                top_tokens=[" once", " happily", " for"],
                top_p_increases=[0.03, 0.31, 0.01]
            ),
            ExplainerInterventionExample(
                prompt="MSI Going Full Throttle @ CeBIT",
                top_tokens=[" Once", " once", " in", " the", " a", " The"],
                top_p_increases=[0.02, 0.01, 0.01, 0.01, 0.01, 0.01]
            ),
        ],
        explanation="storytelling"
    ),
    ExplainerNeuronFormatter(
        intervention_examples=[
            ExplainerInterventionExample(
                prompt="Given 4x is less than 10,",
                top_tokens=[" 4", " 10", " 40", " 2"],
                top_p_increases=[0.11, 0.04, 0.02, 0.01]
            ),
            ExplainerInterventionExample(
                prompt="For some reason",
                top_tokens=[" one", " 1", " fr"],
                top_p_increases=[0.14, 0.01, 0.01]
            ),
            ExplainerInterventionExample(
                prompt="insurance does not cover claims for accounts with",
                top_tokens=[" one", " more", " 10"],
                top_p_increases=[0.10, 0.02, 0.01]
            )
        ],
        explanation="numbers"
    )
]

neuron_prompter = copy.deepcopy(fs_examples[0])
neuron_prompter.explanation = None
print(get_explainer_prompt(neuron_prompter, fs_examples))

scorer_separator = "<PASSAGE>"

def get_scorer_simplicity_prompt(explanation):
    prefix = "Explanation\n\n"
    return f"{prefix}{explanation}{scorer_tokenizer.eos_token}", prefix

def get_scorer_surprisal_prompt(prompt, explanation, few_shot_prompts=None, few_shot_explanations=None, few_shot_generations=None):
    if few_shot_explanations is not None:
        assert few_shot_generations is not None and few_shot_prompts is not None
        assert len(few_shot_explanations) == len(few_shot_generations) == len(few_shot_prompts)
        few_shot_prompt = "\n\n".join(get_scorer_surprisal_prompt(pr, expl) + generation for pr, expl, generation in zip(few_shot_prompts, few_shot_explanations, few_shot_generations)) + "\n\n"
    else:
        few_shot_prompt = ""

    if explanation is None:
        return prompt
    return few_shot_prompt + f"{scorer_separator}The following passage was written with an amplified amount of \"{explanation}\":\n{prompt}"

few_shot_prompts = [
    "from west to east, the westmost of the seven",
    "Given 4x is less than 10,",
    "In information theory, the information content, self-information, surprisal, or Shannon information is a basic quantity derived",
    "My favorite food is",
]
few_shot_explanations = [
    "Asia",
    "numbers",
    "she/her pronouns",
    "fruits and vegetables",
]
few_shot_generations = [
    " wonders of the world is the great wall of china",
    " 4",
    " by her when she was a student at Windsor",
    " oranges",
]
print(get_scorer_surprisal_prompt(few_shot_prompts[0], few_shot_explanations[0], few_shot_prompts, few_shot_explanations, few_shot_generations))

n_intervention_tokens = 5
scorer_intervention_strengths = [0, 10, 32, 100, 320, 1000]
explainer_intervention_strength = 32

path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-9b-pt-res",
    filename=f"layer_{feat_layer}/width_131k/average_l0_51/params.npz",
    force_download=False,
)

params = np.load(path_to_params)
pt_params = {k: torch.from_numpy(v).to(subject_device) for k, v in params.items()}


def get_encoder_decoder_weights(feat_idx, device, random_resid_direction):
    encoder_feat = pt_params["W_enc"][feat_idx, :]
    decoder_feat = pt_params["W_dec"][feat_idx, :]
    if random_resid_direction:
        decoder_feat = torch.randn_like(decoder_feat)
    return encoder_feat, decoder_feat


def garbage_collect():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("CUDA garbage collection performed.")

import torch.nn as nn
class JumpReLUSAE(nn.Module):
  def __init__(self, d_model, d_sae):
    # Note that we initialise these to zeros because we're loading in pre-trained weights.
    # If you want to train your own SAEs then we recommend using blah
    super().__init__()
    self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
    self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
    self.threshold = nn.Parameter(torch.zeros(d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.b_dec = nn.Parameter(torch.zeros(d_model))

  def encode(self, input_acts):
    pre_acts = input_acts @ self.W_enc + self.b_enc
    mask = (pre_acts > self.threshold)
    acts = mask * torch.nn.functional.relu(pre_acts)
    return acts

  def decode(self, acts):
    return acts @ self.W_dec + self.b_dec

  def forward(self, acts):
    acts = self.encode(acts)
    recon = self.decode(acts)
    return recon
  
sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1]).to(subject_device)
sae.load_state_dict(pt_params)
from functools import partial
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
def get_activating_text(example):
    idxs = (example.activations > 0).nonzero(as_tuple=True)[0]
    idx = random.choice(idxs).item()
    act = example.activations[idx]
    return subject_tokenizer.decode(example.tokens[:idx]), act.item()
from tqdm.auto import tqdm
from itertools import product, islice
import time
from collections import Counter
import pandas as pd

random_resid_direction = False  # this is a random baseline
save_path = f"counterfactual_results/generative_{subject_name.split('/')[-1]}_{feat_layer}layer_{n_feats}feats{'_random_dir' if random_resid_direction else ''}.json"
all_results = []
n_explanations = 1
max_generation_length = 16


for iter, record in enumerate(tqdm(loader)):
    garbage_collect()
    
    feat_idx = record.feature.feature_index
    encoder_feat, decoder_feat = get_encoder_decoder_weights(feat_idx, subject_device, random_resid_direction)

    # Remove any hooks
    for l in range(len(subject_layers)):
        subject_layers[l]._forward_hooks.clear()
    

    random.shuffle(record.train)
    scorer_examples = [get_activating_text(e) for e in record.train[:n_test]]
    explainer_examples = [get_activating_text(e) for e in record.train[:n_train]]
    
    # get explanation
    print("Getting explanations...")
    def get_subject_logits(text, layer, intervention_strength=0.0, position=-1, feat=None):
        for l in range(len(subject_layers)):
            subject_layers[l]._forward_hooks.clear()
        subject_layers[layer].register_forward_hook(partial(addition_intervention, intervention_strength=intervention_strength, position=-1, feat=feat))

        inputs = subject_tokenizer(text, return_tensors="pt", add_special_tokens=True).to(subject_device)
        with torch.inference_mode():
            outputs = subject(**inputs)

        return outputs.logits[0, -1, :]
    
    def generate_with_intervention(text, layer, clamp_value=0.0, feat_idx=None):
        for l in range(len(subject_layers)):
            subject_layers[l]._forward_hooks.clear()
        
        inputs = subject_tokenizer(text, return_tensors="pt", add_special_tokens=True).to(subject_device)
        # x[:, slice(None), :] is equivalent to x[:, :, :]
        subject_layers[layer].register_forward_hook(partial(clamping_intervention, clamp_value=clamp_value, feat_idx=feat_idx))
        with torch.inference_mode():
            out = subject.generate(
                **inputs,
                max_new_tokens=max_generation_length,
                num_return_sequences=1,
                temperature=1.0,
                do_sample=True,
            )

        return subject_tokenizer.decode(out[0])

    explainer_time = time.time()
    intervention_examples = []
    for text, act in explainer_examples:
        clean_logits = get_subject_logits(text, feat_layer, intervention_strength=0.0, feat=decoder_feat)
        intervened_logits = get_subject_logits(text, feat_layer, intervention_strength=explainer_intervention_strength, feat=decoder_feat)
        top_probs = (intervened_logits.softmax(dim=-1) - clean_logits.softmax(dim=-1)).topk(n_intervention_tokens)
        
        top_tokens = [subject_tokenizer.decode(i) for i in top_probs.indices]
        top_p_increases = top_probs.values.tolist()
        intervention_examples.append(
            ExplainerInterventionExample(
                prompt=text,
                top_tokens=top_tokens,
                top_p_increases=top_p_increases
            )
        )

    neuron_prompter = ExplainerNeuronFormatter(
        intervention_examples=intervention_examples
    )

    # TODO: improve the few-shot examples
    explainer_prompt = get_explainer_prompt(neuron_prompter, fs_examples)
    explainer_input_ids = explainer_tokenizer(explainer_prompt, return_tensors="pt").input_ids.to(explainer_device)
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
    explainer_time = time.time() - explainer_time
    print(f"Explainer took {explainer_time:.2f} seconds")

    for ie in intervention_examples:
        print(ie.top_tokens)
        print(ie.top_p_increases)
    print(explanations)

    scoring_time = time.time()

    completions = []
    for text, act in scorer_examples:            
        # get generation with and without intervention
        completions.append({"text": text, "act": act, "completions": dict()})
        for name, strength in [("clean", 0), ("intervened", act)]:
            completions[-1]["completions"][name] = generate_with_intervention(text, feat_layer, clamp_value=strength, feat_idx=feat_idx)

    surprisals_by_explanation = dict()
    completions_by_explanation = dict()
    delta_conditional_entropy_by_explanation = dict()
    delta_conditional_entropy_sems_by_explanation = dict()
    delta_entropy_update_by_explanation = dict()
    delta_entropy_update_sems_by_explanation = dict()
    for explanation in explanations:
        surprisals = {"conditional": {"clean": [], "intervened": []}, "unconditional": {"clean": [], "intervened": []}}

        for conditional_name, expl in [("conditional", explanation), ("unconditional", None)]:
            for i, (text, act) in enumerate(scorer_examples):
                
                # get the explanation predictiveness
                scorer_surprisal_prompt = get_scorer_surprisal_prompt(text, expl, few_shot_prompts, few_shot_explanations, few_shot_generations)
                scorer_prompt_ids = scorer_tokenizer(scorer_surprisal_prompt, return_tensors="pt").input_ids.to(scorer_device)
                
                with torch.inference_mode():
                    kv_cache = scorer(scorer_prompt_ids, use_cache=True, return_dict=True).past_key_values
                
                for name, completion in completions[i]["completions"].items():
                    scorer_completion_ids = scorer_tokenizer(completion, return_tensors="pt").input_ids.to(scorer_device)
                    scorer_input_ids = torch.cat([scorer_prompt_ids, scorer_completion_ids], dim=1)
                    labels = scorer_input_ids.clone()
                    labels[:, :scorer_prompt_ids.shape[1]] = -100
                    with torch.inference_mode():
                        out = scorer(scorer_input_ids, labels=labels, return_dict=True, past_key_values=kv_cache)
                        # HF averages over the sequence length, so we undo that
                        surprisals[conditional_name][name].append(out.loss.item() * (scorer_input_ids.shape[1] - scorer_prompt_ids.shape[1]))
                
                del kv_cache


        surprisals = {k: {k2: np.array(v2) for k2, v2 in v.items()} for k, v in surprisals.items()}
        surprisals_by_explanation[explanation] = surprisals
        conditional_surprisals = {k: v for k, v in surprisals["conditional"].items()}
        delta_conditional_entropy_by_explanation[explanation] = (conditional_surprisals["intervened"] - conditional_surprisals["clean"]).mean()
        delta_conditional_entropy_sems_by_explanation[explanation] = (conditional_surprisals["intervened"] - conditional_surprisals["clean"]).std(ddof=1) / np.sqrt(len(conditional_surprisals["intervened"]))
        surprisal_updates = dict()
        for k in surprisals["conditional"]:
            surprisal_updates[k] = surprisals["conditional"][k] - surprisals["unconditional"][k]
        delta_entropy_update_by_explanation[explanation] = (surprisal_updates["intervened"] - surprisal_updates["clean"]).mean()
        delta_entropy_update_sems_by_explanation[explanation] = (surprisal_updates["intervened"] - surprisal_updates["clean"]).std(ddof=1) / np.sqrt(len(surprisal_updates["intervened"]))

    scoring_time = time.time() - scoring_time
    print(f"Scoring took {scoring_time:.2f} seconds")

    print(delta_conditional_entropy_by_explanation)
    print()
    print()
    all_results.append({
        "feat_idx": feat_idx,
        "feat_layer": feat_layer,
        "explanations": dict(explanations),
        "surprisals_by_explanation": surprisals_by_explanation,
        "delta_conditional_entropy_by_explanation": delta_conditional_entropy_by_explanation,
        "delta_conditional_entropy_sems_by_explanation": delta_conditional_entropy_sems_by_explanation,
        "delta_entropy_update_by_explanation": delta_entropy_update_by_explanation,
        "delta_entropy_update_sems_by_explanation": delta_entropy_update_sems_by_explanation,
        "max_delta_conditional_entropy": max(delta_conditional_entropy_by_explanation.values()),
        "max_delta_entropy_update": max(delta_entropy_update_by_explanation.values()),
        "explainer_prompts": [example.prompt for example in intervention_examples],
        "explainer_intervention_strength": explainer_intervention_strength,
        "scorer_examples": scorer_examples,
        "completions": completions,
        "explainer_examples": explainer_examples,
        "neuron_prompter": neuron_prompter,
    })
    if (iter - 1) % 10 == 0:
        all_df = pd.DataFrame(all_results)
        all_df = all_df.sort_values("max_delta_conditional_entropy", ascending=False)
        all_df.to_json(save_path)
all_df = pd.DataFrame(all_results)
all_df = all_df.sort_values("max_delta_conditional_entropy", ascending=False)
all_df.to_json(save_path)
