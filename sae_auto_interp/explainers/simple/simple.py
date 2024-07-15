import re
from typing import List

from .prompts import create_prompt
from ..explainer import (
    Explainer, 
    ExplainerInput
)


class SimpleExplainer(Explainer):
    """
    The Simple explainer generates an explanation using few shot examples
    using just the tokens that are activated in the sequence.
    """

    name = "simple"

    def __init__(
        self,
        client,
        tokenizer,
        max_tokens=200,
        temperature=0.0,
    ):
        self.client = client
        self.tokenizer = tokenizer

        self.max_tokens = max_tokens
        self.temperature = temperature
        
    async def __call__(
        self,
        explainer_in: ExplainerInput,
    ):

        messages = self.build_prompt(
            explainer_in.train_examples,
            explainer_in.record.max_activation
        )
        response = await self.client.generate(
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        explanation = self.parse_explanation(response)

        return explanation
    
    def parse_explanation(self, text:str):
        pattern = r'\[EXPLANATION\]:\s*(.*)'

        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            explanation = match.group(1).strip()
            return explanation
        else:
            return "Explanation not found"
    
    def build_prompt(self,examples,max_activation:float):

        #I think this can be prettier
        str_tokens = []
        activations = []
        sentences= []
        for example in examples:
            str_toks = self.tokenizer.batch_decode(example.tokens)
            str_tokens.append(str_toks)
            activations.append(example.activations)
            sentences.append("".join(str_toks))
            
        undiscovered_feature = ""
        for i in range(len(sentences)):
            decoded = sentences[i]
            activated_tokens,scores = str_tokens[i],activations[i]
            undiscovered_feature += self.formulate_question(i+1,decoded,activated_tokens,scores,max_activation)
        msg = create_prompt(undiscovered_feature)
        return msg
    
    def formulate_question(self,index:int,document:str,activating_tokens:List[str],activations:List[int],max_activation:float) -> str:
        if index == 1:
            question = "Neuron\n"
        else:
            question = ""
        question += f"Document {index}:\n{document}\n"
        question += "Activating tokens:"
        for token,score in zip(activating_tokens,activations):
            if score > 0:
                score = score/max_activation*10
                score = score.round()
                question += f" {token} ({score}),"
        # Remove the last comma
        question = question[:-1] + ".\n\n"
        return question

    

    

    





                
    