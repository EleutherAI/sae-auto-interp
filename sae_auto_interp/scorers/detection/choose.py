
from typing import List
from llama_cpp import Llama
from sae_auto_interp.scorers.detection.base_prompts import EXPLANATION_SYSTEM, FEW_SHOT_EXAMPLES

def formulate_question(explanation:str,document:str) -> str:
    question = f"Explanation: {explanation}\n"
    question += f"Document:\n {document}\n"
    return question


def prompt_model(question:str) -> str:

    msg = []
    msg.append({"role":"system","content":EXPLANATION_SYSTEM})
    for key in FEW_SHOT_EXAMPLES:
        example = FEW_SHOT_EXAMPLES[key]
        msg.append({"role":"user","content":example["user"]})
        msg.append({"role":"assistant","content":example["assistant"]})
    msg.append({"role":"user","content":question})
    return msg

def recall_prompts(sentences:List[str],explanation:str,tokenizer) -> List[str]:
    prompts = []
    for sentence in sentences:
        question = formulate_question(explanation,sentence)
        msg = prompt_model(question)
        prompt = tokenizer.apply_chat_template(msg,tokenize=False)
        prompts.append(prompt)
    return prompts


def recall_evaluations(answers:List[str],number_true:int) -> int:

    #TODO: This could be more general
    correct = 0
    for i,answer in enumerate(answers):
        if "Yes" in answer and i < number_true:
            correct += 1
        elif "No" in answer and i >= number_true:
            correct += 1
    return correct
        
        
    