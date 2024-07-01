import re
from typing import List

from .prompts import create_prompt
from ..explainer import (
    Explainer, 
    ExplainerInput, 
    ExplainerResult
)
from ... import cot_explainer_config as CONFIG 

class ChainOfThought(Explainer):
    """
    The Chain of Thought explainer generates an explanation
    using a few shot examples to guide the model in its forming an explanation.
    """

    name = "cot"

    def __init__(
        self,
        client
    ):
        self.client = client

    async def __call__(
        self,
        explainer_in: ExplainerInput
    ) -> ExplainerResult:
        """
        Generate an explanation using the Chain of Thought model.

        Args:
            explainer_in (ExplainerInput): Input to the explainer.
        """

        simplified, messages = self.build_prompt(
            explainer_in.train_examples, 
            explainer_in.record.max_activation, 
            explainer_in.record.top_logits
        )

        response = await self.client.async_generate(
            messages, 
            max_tokens=CONFIG.max_tokens,
            temperature=CONFIG.temperature
        )

        explanation = self.parse_explanation(response)

        return explanation

        # return ExplainerResult(
        #     explainer_type=self.name,
        #     prompt=simplified,
        #     response=response,
        #     explanation=explanation
        # )
    
    def parse_explanation(self, text: str) -> str:
        """
        Parses the explanation from the response text.
        """
        pattern = r'\[EXPLANATION\]:\s*(.*)'

        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            explanation = match.group(1).strip()
            return explanation
        else:
            return "EXPLANATION"
    
    def flatten(self, array: List[List[str]]) -> str:
        """
        Flatten a nested list of strings and add quotes around each word.

        Args:
            array (List[List[str]]): Nested list of strings.

        Returns:
            str: Flattened string with quotes around each word.
        """

        # Flatten the nested array
        flattened_list = [item for sublist in array for item in sublist]
        
        # Add quotes around each word
        quoted_list = [f'"{word}"' for word in flattened_list]
        
        # Join the list into a single string of words separated by commas
        result_string = ", ".join(quoted_list)
        
        return result_string
    
    def prepare_example(self, example: str) -> str:
        """
        Prepares a set of examples for the Chain of Thought model.
        Extracts contextual tokens for the model to attend to.

        Args:
            example (Example): Example to prepare.

        Returns:
            str: Delimited string with activating tokens.
            List[str]: List of activating tokens.
            List[str]: List of previous tokens.
            List[str]: List of following tokens.
        """
        delimited_string = ""

        activating = []
        previous = []
        following = []

        pos = 0

        while pos < len(example.tokens):
            if pos + 1 < len(example.tokens) and example.activations[pos + 1] > 0.0:
                delimited_string += example.str_toks[pos]
                previous.append(example.str_toks[pos])
                pos += 1
            elif example.activations[pos] > 0.0:
                delimited_string += CONFIG.l

                seq = ""
                while pos < len(example.tokens) and example.activations[pos] > 0.0:
                    
                    delimited_string += example.str_toks[pos]
                    seq += example.str_toks[pos]
                    pos += 1
                
                activating.append(seq)
                if pos < len(example.tokens):
                    following.append(example.str_toks[pos])

                delimited_string += CONFIG.r
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
                    example
                )
            
            activating.append(act)
            previous.append(prev)
            following.append(follow)
            top_examples.append(delimited_string)
            
        top_examples_str = ""
            
        for i, example in enumerate(top_examples):
            top_examples_str += f"Example {i}: {example}\n"

        activating = self.flatten(activating)
        previous = self.flatten(previous)
        following = self.flatten(following)

        simplified, prompt = create_prompt(
            l=CONFIG.l, 
            r=CONFIG.r, 
            examples=top_examples, 
            top_logits=top_logits,
            activating=activating,
            previous=previous,
            following=following,
            simplifiy=True
        )

        messages = [
            {
                "role" : "user",
                "content" : prompt, 
            }
        ]

        return simplified, messages