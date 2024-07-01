
from typing import List
from llama_cpp import Llama
from sae_auto_interp.scorers.generation.base_prompts import FEW_SHOT_EXAMPLES_POS,EXPLANATION_SYSTEM_POS,EXPLANATION_SYSTEM_NEG,FEW_SHOT_EXAMPLES_NEG
import torch

def formulate_question(explanation:str) -> str:
    question = f"Explanation: {explanation}\n"
    return question

def prompt_model_pos(question:str,llm:Llama) -> str:
    msg = []
    msg.append({"role":"system","content":EXPLANATION_SYSTEM_POS})
    for key in FEW_SHOT_EXAMPLES_POS:
        example = FEW_SHOT_EXAMPLES_POS[key]
        msg.append({"role":"user","content":example["user"]})
        msg.append({"role":"assistant","content":example["assistant"]})
    msg.append({"role":"user","content":question})
    answer = llm.create_chat_completion(msg,max_tokens=400)["choices"][0]["message"]["content"]
    return answer

def prompt_model_neg(question:str,llm:Llama) -> str:
        msg = []
        msg.append({"role":"system","content":EXPLANATION_SYSTEM_NEG})
        for key in FEW_SHOT_EXAMPLES_NEG:
            example = FEW_SHOT_EXAMPLES_NEG[key]
            msg.append({"role":"user","content":example["user"]})
            msg.append({"role":"assistant","content":example["assistant"]})
        msg.append({"role":"user","content":question})
        answer = llm.create_chat_completion(msg,max_tokens=400)["choices"][0]["message"]["content"]
        return answer


def enumerate_evaluations(simulator,explanation):


    question = formulate_question(explanation)
    positive_answers = prompt_model_pos(question,simulator)
    positive_answers = positive_answers.split("\n")
    cleaned_positive_answers = []
    for answer in positive_answers:
        if len(answer) == 0:
            continue
        if answer[0] in ["1","2","3","4","5"]:
            cleaned_positive_answers.append(answer[2:])
    # negative_answers = prompt_model_neg(question,simulator)
    # negative_answers = negative_answers.split("\n")
    # cleaned_negative_answers = []
    # for answer in negative_answers:
    #     if len(answer) == 0:
    #         continue
    #     if answer[0] in ["1","2","3","4","5"]:
    #         cleaned_negative_answers.append(answer[2:])

    return cleaned_positive_answers#,cleaned_negative_answers
        
def score_evaluations(autoencoder,tokenizer,sentence,model,feature,layer,pos=True):
    score = 0
    encoded = tokenizer(sentence,return_tensors="pt").input_ids
    _,model_acts = model.run_with_cache(encoded, remove_batch_dim=False)
    layer_acts = model_acts[f"blocks.{layer}.hook_resid_post"]
    features = autoencoder.encode(layer_acts)[0]
    indices = torch.nonzero(features.abs()>1e-5)
    if pos:
        if feature in indices[:,2]:
            score += 1
    else:
        if feature not in indices[:,2]:
            score += 1
    return score
    