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
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    def __call__(
        self,
        explainer_in: ExplainerInput,
        echo=False
    ) -> ExplainerResult:
        simplified, user_prompt = self.build_prompt(
            explainer_in.train_examples, 
            self.tokenizer
        )

        prompt = [
            {
                "role" : "user",
                "content" : user_prompt, 
            }
        ]

        generation_args = {
            "max_tokens" : simple_cfg.max_tokens,
            "temperature" : simple_cfg.temperature,
        }

        response = self.client.generate(prompt, generation_args)
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

        tokens = []
        activations = []
        for example in examples:
            tokens.append(example.tokens)
            activations.append(example.activations)
        #this is broken
        sentences,token_score_pairs = self.template_explanation(tokens,activations,tokenizer)
        undiscovered_feature = ""
        for i in range(len(sentences)):
            decoded = sentences[i]
            activated_tokens,scores = token_score_pairs[i]
            undiscovered_feature += self.formulate_question(i+1,decoded,activated_tokens,scores)
        prompt = self.make_prompt(undiscovered_feature)
        spelled_out = tokenizer.apply_chat_template(prompt,add_generation_prompt=True,tokenize=False)
        return prompt,spelled_out
    
    def formulate_question(self,index:int,document:str,activating_tokens:List[str],activations:List[int]) -> str:
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

    def make_prompt(self,question:str) -> str:

        msg = []
        msg.append({"role":"system","content":EXPLANATION_SYSTEM})
        for key in FEW_SHOT_EXAMPLES:
            example = FEW_SHOT_EXAMPLES[key]
            msg.append({"role":"user","content":example["user"]})
            msg.append({"role":"assistant","content":example["assistant"]})
        msg.append({"role":"user","content":question})
        return msg

    #TODO: This should be moved somewhere else
    def template_explanation(self,top_sentences,top_activations,top_indices,tokenizer,number_examples=10):
        
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

    





                
    


