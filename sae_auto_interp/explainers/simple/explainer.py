import torch
from torch import Tensor
from sae_auto_interp.explainers.simple.prompts import EXPLANATION_SYSTEM, FEW_SHOT_EXAMPLES
from typing import List

from transformers import AutoTokenizer
import numpy as np

import re


from ..explainer import Explainer, ExplainerInput, ExplainerResult
from ...clients.api import get_client
from ... import simple_cfg

class SimpleExplainer(Explainer):
    
    def __init__(
        self,
        model,
        provider,
    ):
        self.name = "simple"
        self.client = get_client(provider, model)
        #TODO: Monkeypatch
        #self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    def __call__(
        self,
        explainer_in: ExplainerInput,
        echo=False
    ) -> ExplainerResult:
        simplified, user_prompt = self.build_prompt(
            explainer_in.train_examples,
        )

        # prompt = [
        #     {
        #         "role" : "user",
        #         "content" : user_prompt, 
        #     }
        # ]

        generation_args = {
            "max_tokens" : simple_cfg.max_tokens,
            "temperature" : simple_cfg.temperature,
        }
        
        response = self.client.generate(user_prompt, generation_args)
        explanation = self.parse_explanation(response)

        return ExplainerResult(
            explainer_type=self.name,
            input=simplified,
            response=response,
            explanation=explanation
        )
    
    def parse_explanation(self, text:str):
        pattern = r'Explanation:\s*(.*)'

        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            # Get the explanation from the Regex object and strip
            # Surrounding whitespace
            explanation = match.group(1).strip()
            return explanation
        else:
            return "Explanation:"
    
    def build_prompt(self,examples,tokenizer):

        #I think this can be prettier
        tokens = []
        activations = []
        sentences= []
        for example in examples:
            tokens.append(example.tokens)
            activations.append(example.activations)
            sentences.append(example.text)
        max_activation = max([max(activation) for activation in activations])
        undiscovered_feature = ""
        for i in range(len(sentences)):
            decoded = sentences[i]
            activated_tokens,scores = tokens[i],activations[i]
            undiscovered_feature += self.formulate_question(i+1,decoded,activated_tokens,scores,max_activation)
        prompt,question = self.make_prompt(undiscovered_feature)
        spelled_out = tokenizer.apply_chat_template(prompt,add_generation_prompt=True,tokenize=False)
        return question,spelled_out
    
    def formulate_question(self,index:int,document:str,activating_tokens:List[str],activations:List[int],max_activation:float) -> str:
        if index == 1:
            question = "Neuron\n"
        else:
            question = ""
        question += f"Document {index}:\n{document}\n"
        question += "Activating tokens:"
        for token,score in zip(activating_tokens,activations):
            if score > 0:
                score = round(score/max_activation*10,0)
                question += f" {token} ({score}),"
        # Remove the last comma
        question = question[:-1] + ".\n\n"
        return question

    def make_prompt(self,question:str) -> str:

        msg = []
        msg.append({"role":"system","content":EXPLANATION_SYSTEM})
        for key in FEW_SHOT_EXAMPLES:
            example = FEW_SHOT_EXAMPLES[key]
            msg.append({"role":"user","content":example["user"]})
            msg.append({"role":"assistant","content":example["assistant"]})
        msg.append({"role":"user","content":question})
        return msg,question

    

    





                
    


