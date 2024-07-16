import re
from typing import List

from .prompt_builder import build_prompt
from ..explainer import (
    Explainer, 
    ExplainerInput, 
)

class SimpleExplainer(Explainer):
    name = "Simple"
    def __init__(
        self,
        client,
        tokenizer,
        cot: bool = False,
        logits: bool = False,
        activations: bool = False,
        max_tokens:int =200,
        temperature:float =0.0,
        threshold:float =0.6,
        echo: bool = False
    ):
        self.client = client
        self.tokenizer = tokenizer

        self.cot = cot
        self.logits = logits
        self.activations = activations
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.threshold = threshold

        self.echo = echo

    async def __call__(
        self,
        explainer_in: ExplainerInput
    ):
        
        if self.logits:
            messages = self._build_prompt(
                explainer_in.train_examples,
                explainer_in.record.top_logits
            )
        else:
            messages = self._build_prompt(
                explainer_in.train_examples,
                None
            )
        response = await self.client.generate(
            messages, 
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        explanation = self.parse_explanation(response)

        if self.echo:
            return [messages[-1], response], explanation
        return explanation

    def parse_explanation(self, text: str) -> str:
        match = re.search(
            r'\[EXPLANATION\]:\s*(.*)', 
            text, re.DOTALL
        )
        
        return match.group(1).strip() \
            if match else "Explanation could not be parsed."
        
    def _highlight(self, index, example):
        result = f"Example {index}: "

        threshold = example.max_activation * self.threshold
        str_toks = example.decode(self.tokenizer)
        activations = example.activations

        check = lambda i: activations[i] > threshold 

        i = 0
        while i < len(str_toks):
            if check(i):
                result += "<<"

                while (
                    i < len(str_toks) 
                    and check(i)
                ):
                    result += str_toks[i]
                    i += 1
                result += ">>"
            else:
                result += str_toks[i]
                i += 1

        return "".join(result)

    def _build_prompt(self, examples, top_logits):
        
        highlighted_examples = []

        for i, example in enumerate(examples):
            highlighted_examples.append(
                self._highlight(
                    i + 1,
                    example
                )
            )
            
            if self.activations:
                for i, activation in enumerate(example.normalized_activations):
                    if activation > example.max_activation * self.threshold:
                        highlighted_examples.append(
                            f"Activation {example.str_toks[i]}:{activation}, "
                        )
            
        highlighted_examples = "\n".join(highlighted_examples)

        return build_prompt(
            examples=highlighted_examples,
            cot= self.cot,
            activations= self.activations,
            top_logits= top_logits,
        )

