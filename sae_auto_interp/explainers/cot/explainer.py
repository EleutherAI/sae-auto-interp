

from ..explainer import Explainer, ExplainerInput, ExplainerResult
from .prompts import create_prompt

import re

from ... import example_cfg as cfg
from ... import cot_cfg

from ...clients.api import get_client


class ChainOfThought(Explainer):

    def __init__(
        self,
        model,
        provider,
    ):
        self.name = "cot"
        self.client = get_client(provider, model)

    def __call__(
        self,
        explainer_in: ExplainerInput,
        echo=False
    ) -> ExplainerResult:
        simplified, user_prompt = self.build_prompt(
            explainer_in.train_examples, 
            explainer_in.record.max_activation, 
            explainer_in.record.top_logits
        )

        prompt = [
            {
                "role" : "user",
                "content" : user_prompt, 
            }
        ]

        generation_args = {
            "max_tokens" : cot_cfg.max_tokens,
            "temperature" : cot_cfg.temperature,
        }

        response = self.client.generate(prompt, generation_args)
        explanation = self.parse_explanation(response)

        return ExplainerResult(
            explainer_type=self.name,
            input=simplified,
            response=response,
            explanation=explanation
        )
    
    def parse_explanation(self, text):
        pattern = r'\[EXPLANATION\]:\s*(.*)'

        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            # Get the explanation from the Regex object and strip
            # Surrounding whitespace
            explanation = match.group(1).strip()
            return explanation
        else:
            return "EXPLANATION"
    
    def flatten_and_join_with_quotes(self, array):
        # Flatten the nested array
        flattened_list = [item for sublist in array for item in sublist]
        
        # Add quotes around each word
        quoted_list = [f'"{word}"' for word in flattened_list]
        
        # Join the list into a single string of words separated by commas
        result_string = ", ".join(quoted_list)
        
        return result_string
    
    def prepare_example(self, example, max_act):
        delimited_string = ""
        activation_threshold = 0.0

        activating = []
        previous = []
        following = []

        pos = 0
        while pos < len(example.tokens):
            if pos + 1 < len(example.tokens) and example.activations[pos + 1] > activation_threshold:
                delimited_string += example.str_toks[pos]
                previous.append(example.str_toks[pos])
                pos += 1
            elif example.activations[pos] > activation_threshold:
                delimited_string += cfg.l

                seq = ""
                while pos < len(example.tokens) and example.activations[pos] > activation_threshold:
                    
                    delimited_string += example.str_toks[pos]
                    seq += example.str_toks[pos]
                    pos += 1
                
                activating.append(seq)
                if pos < len(example.tokens):
                    following.append(example.str_toks[pos])

                delimited_string += cfg.r
            else:
                delimited_string += example.str_toks[pos]
                pos += 1

        return delimited_string, activating, previous, following

    def build_prompt(self, examples, max_act, top_logits):

        activating = []
        previous = []
        following = []
        top_examples = []

        for example in examples:
            delimited_string, act, prev, follow = \
                self.prepare_example(
                    example, 
                    max_act,
                )
            
            activating.append(act)
            previous.append(prev)
            following.append(follow)
            top_examples.append(delimited_string)
            
        top_examples_str = ""
            
        for i, example in enumerate(top_examples):
            top_examples_str += f"Example {i}: {example}\n"

        activating = self.flatten_and_join_with_quotes(activating)
        previous = self.flatten_and_join_with_quotes(previous)
        following = self.flatten_and_join_with_quotes(following)

        simplified, user_prompt = create_prompt(
            l=cfg.l, 
            r=cfg.r, 
            examples=top_examples, 
            top_logits=top_logits,
            activating=activating,
            previous=previous,
            following=following,
            simplifiy=True
        )


        return simplified, user_prompt