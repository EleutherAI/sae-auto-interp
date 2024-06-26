import torch

from sae_auto_interp.explanations.base_prompts import EXPLANATION_SYSTEM, FEW_SHOT_EXAMPLES
from typing import List
from llama_cpp import Llama

import numpy as np


def template_explanation(top_sentences,top_activations,top_indices,tokenizer,number_examples=10):
    
    max_activation = top_activations.max().item()
    
    selection_indices = np.random.choice(top_sentences.shape[0],number_examples,replace=False)
    sentences = []
    token_score_pairs = []
    for i in selection_indices:
        sentence = top_sentences[i]
        activated = torch.nonzero(top_indices[:,0] == i)
        activated_indices = top_indices[activated]
        scores = torch.zeros(sentence.shape[0])
    
        for j in range(activated.shape[0]):
            
            activating_token = activated_indices[j,:,1].int()
            score = top_activations[activated[j]].item()
            score = score/max_activation*10
            scores[activating_token] = score
        max_score_index = scores.argmax()
        start = max(max_score_index-20,0)
        end = min(max_score_index+10,sentence.shape[0])
        tokens = []
        for token in sentence[start:end]:
            tokens.append(tokenizer.decode(token))

        token_score_pairs.append((tokens,scores[start:end]))
        sentences.append(tokenizer.decode(sentence[start:end]))

    max_activation = top_activations.max().item()
    
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
        if score > 0:
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

                
    


