import torch
from torch import Tensor

from sae_auto_interp.explanations.base_prompts import EXPLANATION_SYSTEM, FEW_SHOT_EXAMPLES
from typing import List
from llama_cpp import Llama

import numpy as np


def template_explanation(top_sentences,top_activations,top_indices,tokenizer,number_examples=10):
    selection_indices = np.random.choice(top_sentences.shape[0],number_examples,replace=False)
    sentences = []
    token_score_pairs = []
    for i in selection_indices:
        sentence = top_sentences[i]
        decoded = tokenizer.decode(sentence,skip_special_tokens=True)
        sentences.append(decoded)
        activated = torch.nonzero(top_indices[:,0] == i)
        activated_indices = top_indices[activated]
        activated_tokens = []
        scores = []
        for j in range(activated.shape[0]):
            
            activating_token = activated_indices[j,:,1]

            activating_token = tokenizer.decode(sentence[activating_token.int()],skip_special_tokens=True)
            score = top_activations[activated[j]].item()
            score = int(round(score,0))
            activated_tokens.append(activating_token)
            scores.append(score)
            
        token_score_pairs.append((activated_tokens,scores))
    return sentences,token_score_pairs
    
    

def generate_explanation(explainer_model,sentences,token_score_pairs):
    
    undiscovered_feature = ""
    for i in range(len(sentences)):
        decoded = sentences[i]
        activated_tokens,scores = token_score_pairs[i]
        undiscovered_feature += formulate_question(i+1,decoded,activated_tokens,scores)
    answer = prompt_model(undiscovered_feature,explainer_model)
    return answer
    




def formulate_question(index:int,document:str,activating_tokens:List[str],activations:List[int]) -> str:
    if index == 1:
        question = "Neuron\n"
    else:
        question = ""
    question += f"Document {index}:\n{document}\n"
    question += "Activating tokens:"
    for token,score in zip(activating_tokens,activations):
        question += f" {token} ({score}),"
    # Remove the last comma
    question = question[:-1] + ".\n\n"
    return question

def prompt_model(question:str,llm:Llama) -> str:

    msg = []
    msg.append({"role":"system","content":EXPLANATION_SYSTEM})
    for key in FEW_SHOT_EXAMPLES:
        example = FEW_SHOT_EXAMPLES[key]
        msg.append({"role":"user","content":example["user"]})
        msg.append({"role":"assistant","content":example["assistant"]})
    msg.append({"role":"user","content":question})
    answer = llm.create_chat_completion(msg,stop=".",max_tokens=100)["choices"][0]["message"]["content"]
    return answer

                
    


