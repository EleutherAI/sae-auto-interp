
from typing import List
from llama_cpp import Llama
from sae_auto_interp.evaluations.choose.base_prompts import EXPLANATION_SYSTEM, FEW_SHOT_EXAMPLES


def formulate_question(explanation:str,document:str) -> str:
    question = f"Explanation: {explanation}\n"
    question += f"Document:\n {document}\n"
    return question


def prompt_model(question:str,llm:Llama) -> str:

    msg = []
    msg.append({"role":"system","content":EXPLANATION_SYSTEM})
    for key in FEW_SHOT_EXAMPLES:
        example = FEW_SHOT_EXAMPLES[key]
        msg.append({"role":"user","content":example["user"]})
        msg.append({"role":"assistant","content":example["assistant"]})
    msg.append({"role":"user","content":question})
    answer = llm.create_chat_completion(msg,stop=".",max_tokens=5)["choices"][0]["message"]["content"]
    return answer


def recall_evaluations(simulator,sentences,explanation):

    #TODO: This could be more general
    correct = 0
    for i,sentence in enumerate(sentences):
        question = formulate_question(explanation,sentence)
        answer = prompt_model(question,simulator)
        if "Yes" in answer and i < 5:
            correct += 1
        elif "No" in answer and i >= 5:
            correct += 1
    return correct
        
        
    