# load subject model
# load SAEs without attaching them to the model
# for now just use the Islam feature and explanation
# load a scorer. The prompt should have the input as well this time
# (for now) on random pretraining data, evaluate gpt2 with a hook that 
# adds a multiple of the Islam feature to the appropriate residual stream layer and position
# Get the pre- and post-intervention output distributions of gpt2
# (TODO: check if all the Islam features just have similar embeddings)
# Show this to the scorer and get a score (scorer should be able to have a good prior without being given the clean output distribution)
# Also get a simplicity score for the explanation
import pandas as pd
from pathlib import Path
import json

import random

with open("pile.jsonl", "r") as f:
    pile = random.sample([json.loads(line) for line in f.readlines()], 100000)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

subject_device = "cuda:0"

subject_name = "EleutherAI/pythia-70m-deduped"
subject = AutoModelForCausalLM.from_pretrained(subject_name).to(subject_device)
subject_tokenizer = AutoTokenizer.from_pretrained(subject_name)
subject_tokenizer.pad_token = subject_tokenizer.eos_token
subject.config.pad_token_id = subject_tokenizer.eos_token_id
from transformers import BitsAndBytesConfig

scorer_device = "cuda:1"
scorer_name = "meta-llama/Meta-Llama-3.1-8B"
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
    # ExplainerNeuronFormatter(
    #     intervention_examples=[
    #         ExplainerInterventionExample(
    #             prompt="He owned the watch for a long time. While he never said it was",
    #             top_tokens=[" hers", " hers", " hers"],
    #             top_p_increases=[0.09, 0.06, 0.06, 0.5]
    #         ),
    #         ExplainerInterventionExample(
    #             prompt="For some reason",
    #             top_tokens=[" she", " her", " hers"],
    #             top_p_increases=[0.14, 0.01, 0.01]
    #         ),
    #         ExplainerInterventionExample(
    #             prompt="insurance does not cover",
    #             top_tokens=[" her", " women", " her's"],
    #             top_p_increases=[0.10, 0.02, 0.01]
    #         )
    #     ],
    #     explanation="she/her pronouns"
    # )
]

neuron_prompter = copy.deepcopy(fs_examples[0])
neuron_prompter.explanation = None
print(get_explainer_prompt(neuron_prompter, fs_examples))

def get_scorer_simplicity_prompt(explanation):
    prefix = "Explanation\n\n"
    return f"{prefix}{explanation}{scorer_tokenizer.eos_token}", prefix

def get_scorer_predictiveness_prompt(prompt, explanation, few_shot_prompts=None, few_shot_explanations=None, few_shot_tokens=None):
    if few_shot_explanations is not None:
        assert few_shot_tokens is not None and few_shot_prompts is not None
        assert len(few_shot_explanations) == len(few_shot_tokens) == len(few_shot_prompts)
        few_shot_prompt = "\n\n".join(get_scorer_predictiveness_prompt(pr, expl) + token for pr, expl, token in zip(few_shot_prompts, few_shot_explanations, few_shot_tokens)) + "\n\n"
    else:
        few_shot_prompt = ""
    return few_shot_prompt + f"Explanation: {explanation}\n<PROMPT>{prompt}</PROMPT>"

few_shot_prompts = ["My favorite food is", "From west to east, the westmost of the seven"]
few_shot_explanations = ["fruits and vegetables", "ateg"]
few_shot_tokens = [" oranges", "WAY"]
# few_shot_prompts = ["My favorite food is", "From west to east, the westmost of the seven", "He owned the watch for a long time. While he never said it was"]
# few_shot_explanations = ["fruits and vegetables", "ateg", "she/her pronouns"]
# few_shot_tokens = [" oranges", "WAY", " hers"]
print(get_scorer_predictiveness_prompt(few_shot_prompts[0], few_shot_explanations[0], few_shot_prompts, few_shot_explanations, few_shot_tokens))
from functools import partial

def intervene(module, input, output, intervention_strength=10.0, position=-1, feat=None):
    hiddens = output[0]  # the later elements of the tuple are the key value cache
    hiddens[:, position, :] += intervention_strength * feat.to(hiddens.device)  # type: ignore

def get_texts(n, seed=42, randomize_length=True):
    random.seed(seed)
    texts = []
    for _ in range(n):
        
        # sample a random text from the pile, and stop it at a random token position, less than 64 tokens
        text = random.choice(pile)["text"]
        tokenized_text = subject_tokenizer.encode(text, add_special_tokens=False, max_length=64, truncation=True)
        if len(tokenized_text) < 1:
            continue
        if randomize_length:
            stop_pos = random.randint(1, min(len(tokenized_text), 63))
        else:
            stop_pos = 63
        text = subject_tokenizer.decode(tokenized_text[:stop_pos])
        texts.append(text)
    return texts

n_explainer_texts = 3
n_scorer_texts = 3
n_explanations = 5
# explainer_texts = get_texts(n_explainer_texts)
# explainer_texts = ["Current religion:", "A country that is", "Many people believe that"]
# scorer_texts = get_texts(n_scorer_texts)
scorer_vocab = scorer_tokenizer.get_vocab()
subject_vocab = subject_tokenizer.get_vocab()

# Pre-compute the mapping of subject tokens to scorer tokens
subject_to_scorer = {}
text_subject_to_scorer = {}
for subj_tok, subj_id in subject_vocab.items():
    if subj_tok in scorer_vocab:
        subject_to_scorer[subj_id] = scorer_vocab[subj_tok]
        text_subject_to_scorer[subj_tok] = subj_tok
    else:
        for i in range(len(subj_tok) - 1, 0, -1):
            if subj_tok[:i] in scorer_vocab:
                subject_to_scorer[subj_id] = scorer_vocab[subj_tok[:i]]
                text_subject_to_scorer[subj_tok] = subj_tok[:i]
                break
        else:
            print(f"No scorer token found for '{subj_tok}'")
            subject_to_scorer[subj_id] = len(scorer_vocab) - 3  # some very rare token
            print(f"Using '{scorer_tokenizer.decode([len(scorer_vocab) - 3])}' as a placeholder for '{subj_tok}'")
subject_ids = torch.tensor(list(subject_to_scorer.keys()), device=scorer_device)
scorer_ids = torch.tensor(list(subject_to_scorer.values()), device=scorer_device)
n_intervention_examples = 5
n_candidate_texts = 500
candidate_texts = get_texts(n_candidate_texts, randomize_length=False)
from collections import Counter
import torch
import numpy as np
from sae_auto_interp.autoencoders.OpenAI.model import Autoencoder
from itertools import product
from tqdm import tqdm
import time

try:
    subject_layers = subject.transformer.h
except:
    subject_layers = subject.gpt_neox.layers

def get_encoder_decoder_weights(feat_idx, feat_layer, device, random_resid_direction=False):
    # weight_dir = "/mnt/ssd-1/gpaulo/SAE-Zoology/weights/gpt2_128k"
    # path = f"{weight_dir}/{feat_layer}.pt"
    # state_dict = torch.load(path)
    # ae = Autoencoder.from_state_dict(state_dict=state_dict)
    # decoder_feat = ae.decoder.weight[:, feat_idx].to(device)
    # encoder_feat = ae.encoder.weight[feat_idx, :].to(device)
    weight_dir = f"/mnt/ssd-1/alexm/dictionary_learning/dictionaries/pythia-70m-deduped/resid_out_layer{feat_layer}/10_32768/ae.pt"
    state_dict = torch.load(weight_dir)
    encoder_feat = state_dict['encoder.weight'][feat_idx, :]
    decoder_feat = state_dict['decoder.weight'][:, feat_idx]
    if random_resid_direction:
        en_norm, de_norm = encoder_feat.norm(), decoder_feat.norm()
        encoder_feat = torch.randn_like(encoder_feat)
        decoder_feat = torch.randn_like(decoder_feat)
        encoder_feat = encoder_feat * en_norm / encoder_feat.norm()
        decoder_feat = decoder_feat * de_norm / decoder_feat.norm()
    return encoder_feat, decoder_feat

all_results = []

random_resid_direction = False  # this is a random baseline
feat_idxs = list(range(1000))
feat_layers = [4, 2]
save_path = f"counterfactual_results/neg_{subject_name.split('/')[-1]}_{len(feat_layers)}layers_{len(feat_idxs)}feats{'_random_dir' if random_resid_direction else ''}.json"
total_iterations = len(feat_idxs) * len(feat_layers)
for iter, (feat_idx, feat_layer) in enumerate(tqdm(product(feat_idxs, feat_layers), total=total_iterations)):
    scorer_intervention_strengths = [0, -10, -32, -100, -320, -1000]
    explainer_intervention_strength = -32

    print("Loading autoencoder...", end="")
    encoder_feat, decoder_feat = get_encoder_decoder_weights(feat_idx, feat_layer, subject_device, random_resid_direction)

    ### Find examples where the feature activates
    # Remove any hooks
    for l in range(len(subject_layers)):
        subject_layers[l]._forward_hooks.clear()
    print("done")

    selection_time = time.time()
    subtexts = []
    subtext_acts = []
    for text in tqdm(candidate_texts, total=len(candidate_texts)):
        input_ids = subject_tokenizer(text, return_tensors="pt").input_ids.to(subject_device)
        with torch.inference_mode():
            out = subject(input_ids, output_hidden_states=True)
            # hidden_states is actually one longer than the number of layers, because it includes the input embeddings
            h = out.hidden_states[feat_layer + 1].squeeze(0)
            # feat_acts = ae.activation(ae.encoder(h))[:, feat_idx]
            feat_acts = h @ encoder_feat
            # the first token position just has way higher norm all the time for some reason
            feat_acts[0] = 0

        for i in range(1, len(feat_acts) + 1):
            reassembled_text = subject_tokenizer.decode(input_ids[0, :i])
            subtexts.append(reassembled_text)
            subtext_acts.append(feat_acts[i - 1].item())

    # get a random sample of activating contexts
    subtext_acts = torch.tensor(subtext_acts)
    candidate_quantile = 0.994
    candidate_indices = subtext_acts.topk(int(len(subtext_acts) * (1 - candidate_quantile))).indices
    sampled_indices = np.random.choice(candidate_indices.numpy(), n_scorer_texts + n_explainer_texts, replace=False)
    
    # Get top k subtexts and their activations
    sampled_subtexts = [subtexts[i] for i in sampled_indices]
    sampled_activations = subtext_acts.numpy()[sampled_indices]

    random.shuffle(sampled_subtexts)  # just as assurance
    scorer_texts = sampled_subtexts[:n_scorer_texts]
    explainer_texts = sampled_subtexts[n_scorer_texts:]
    selection_time = time.time() - selection_time
    print(f"Selection took {selection_time:.2f} seconds")

    # get explanation
    def get_subject_logits(text, layer, intervention_strength=0.0, position=-1, feat=None):
        for l in range(len(subject_layers)):
            subject_layers[l]._forward_hooks.clear()
        subject_layers[layer].register_forward_hook(partial(intervene, intervention_strength=intervention_strength, position=-1, feat=feat))

        inputs = subject_tokenizer(text, return_tensors="pt", add_special_tokens=True).to(subject_device)
        with torch.inference_mode():
            outputs = subject(**inputs)

        return outputs.logits[0, -1, :]

    explainer_time = time.time()
    intervention_examples = []
    for text in explainer_texts:
        clean_logits = get_subject_logits(text, feat_layer, intervention_strength=0.0, feat=decoder_feat)
        intervened_logits = get_subject_logits(text, feat_layer, intervention_strength=explainer_intervention_strength, feat=decoder_feat)
        top_probs = (intervened_logits.softmax(dim=-1) - clean_logits.softmax(dim=-1)).topk(n_intervention_examples)
        
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
        samples = explainer.generate(explainer_input_ids, max_new_tokens=100, eos_token_id=explainer_tokenizer.encode("\n")[-1], num_return_sequences=n_explanations)[:, explainer_input_ids.shape[1]:]
    explanations = Counter([explainer_tokenizer.decode(sample).split("\n")[0].strip() for sample in samples])
    explainer_time = time.time() - explainer_time
    print(f"Explainer took {explainer_time:.2f} seconds")

    for ie in intervention_examples:
        print(ie.top_tokens)
        print(ie.top_p_increases)
    print(explanations)

    scoring_time = time.time()
    predictiveness_score_by_explanation = dict()
    normalized_predictiveness_score_by_explanation = dict()
    all_pred_scores = []
    scoring_interventions = dict()
    for explanation in explanations:
        expl_predictiveness_scores = []
        scoring_interventions[explanation] = dict()
        for scorer_intervention_strength in tqdm(scorer_intervention_strengths):
            
            current_pred_scores = []
            max_intervened_prob = 0.0
            scoring_interventions[explanation][scorer_intervention_strength] = []
            for text in scorer_texts:
                
                intervened_probs = get_subject_logits(text, feat_layer, intervention_strength=scorer_intervention_strength, feat=decoder_feat).softmax(dim=-1).to(scorer_device)

                # get the explanation predictiveness
                scorer_predictiveness_prompt = get_scorer_predictiveness_prompt(text, explanation, few_shot_prompts, few_shot_explanations, few_shot_tokens)
                scorer_input_ids = scorer_tokenizer(scorer_predictiveness_prompt, return_tensors="pt").input_ids.to(scorer_device)
                with torch.inference_mode():
                    scorer_logits = scorer(scorer_input_ids).logits[0, -1, :]
                    scorer_logp = scorer_logits.log_softmax(dim=-1)
                
                current_pred_scores.append((intervened_probs[subject_ids] * scorer_logp[scorer_ids]).sum())

                topk = intervened_probs.topk(n_intervention_examples).indices
                top_tokens = [subject_tokenizer.decode(i) for i in topk]
                scoring_interventions[explanation][scorer_intervention_strength].append({
                    "prompt": text,
                    "top_tokens": top_tokens,
                    "top_token_probs": intervened_probs[topk].tolist()
                })

            expl_predictiveness_scores.append(torch.tensor(current_pred_scores).mean().item())
            all_pred_scores.extend(current_pred_scores * explanations[explanation])  # as if we did the inference on the scorer multiple times
    
        assert scorer_intervention_strengths[0] == 0
        null_predictiveness_score = expl_predictiveness_scores[0]
        normalized_predictiveness_scores = [score - null_predictiveness_score for score in expl_predictiveness_scores[1:]]
        normalized_predictiveness_score = sum(normalized_predictiveness_scores) / len(normalized_predictiveness_scores)
        predictiveness_score = normalized_predictiveness_score + null_predictiveness_score
        
        predictiveness_score_by_explanation[explanation] = predictiveness_score
        normalized_predictiveness_score_by_explanation[explanation] = normalized_predictiveness_score
    
    # note that this computes stderr over explanations, pile samples, *and* intervention strengths (which is kind of weird)
    pred_score_stderr = torch.std(torch.tensor(all_pred_scores)).item() / len(all_pred_scores) ** 0.5
    pred_score = torch.mean(torch.tensor(all_pred_scores)).item()
    normalized_predictiveness_score = sum([normalized_predictiveness_score_by_explanation[explanation] * count for explanation, count in explanations.items()]) / sum(explanations.values())

    scoring_time = time.time() - scoring_time
    print(f"Scoring took {scoring_time:.2f} seconds")

    print(f"{normalized_predictiveness_score=}")
    print()
    print()
    all_results.append({
        "feat_idx": feat_idx,
        "feat_layer": feat_layer,
        "explanations": dict(explanations),
        "predictiveness_score": pred_score,
        "normalized_predictiveness_score": normalized_predictiveness_score,
        "predictiveness_score_stderr": pred_score_stderr,
        "max_predictiveness_score": max(predictiveness_score_by_explanation.values()),
        "max_normalized_predictiveness_score": max(normalized_predictiveness_score_by_explanation.values()),
        "explainer_prompts": [example.prompt for example in intervention_examples],
        "explainer_top_tokens": [example.top_tokens for example in intervention_examples],
        "explainer_top_p_increases": [example.top_p_increases for example in intervention_examples],
        "scorer_intervention_strengths": scorer_intervention_strengths,
        "explainer_intervention_strength": explainer_intervention_strength,
        "scorer_texts": scorer_texts,
        "explainer_texts": explainer_texts,
        "predictiveness_score_by_explanation": predictiveness_score_by_explanation,
        "normalized_predictiveness_score_by_explanation": normalized_predictiveness_score_by_explanation,
        "scoring_interventions": scoring_interventions
    })
    if (iter - 1) % 10 == 0:
        all_df = pd.DataFrame(all_results)
        all_df = all_df.sort_values("predictiveness_score", ascending=False)
        all_df.to_json(save_path)
all_df = pd.DataFrame(all_results)
all_df = all_df.sort_values("predictiveness_score", ascending=False)
all_df.to_json(save_path)
