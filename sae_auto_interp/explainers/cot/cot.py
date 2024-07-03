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

        messages = self.build_prompt(
            explainer_in.train_examples,
            explainer_in.record.top_logits
        )

        response = await self.client.generate(
            messages, 
            max_tokens=CONFIG.max_tokens,
            temperature=CONFIG.temperature
        )

        explanation = self.parse_explanation(response)

        return explanation

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
            return "Explantion could not be parsed."
    
    def flatten(self, array: List[List[str]]) -> str:
        """
        Flatten a nested list of strings and add quotes around each word.

        Args:
            array (List[List[str]]): Nested list of strings.

        Returns:
            str: Flattened string with quotes around each word.
        """

        # Flatten the nested array
        flattened_list = [
            f'"{item}"' 
            for sublist in array 
            for item in sublist
        ]
        
        result_string = ", ".join(flattened_list)
        
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
        """
        delimited_string = ""

        activating = []
        previous = []

        pos = 0

        while pos < len(example.tokens):
            
            current_tok = example.str_toks[pos]
            current_act = example.activations[pos]

            # Check if next token will activate
            if (pos + 1 < len(example.tokens) 
                and example.activations[pos + 1] > 0.0
            ):
                delimited_string += current_tok
                previous.append(current_tok)
                pos += 1

            # Check if current token activates
            elif current_act > 0.0:
                delimited_string += CONFIG.l

                # Build activating token chunk and
                # delimited string at the same time
                seq = ""
                while pos < len(example.tokens) and current_act > 0.0:
                    delimited_string += current_tok
                    seq += current_tok
                    pos += 1
                activating.append(seq)

                delimited_string += CONFIG.r
            
            # Else, keep building the delimited string
            else:
                delimited_string += current_tok
                pos += 1

        return delimited_string, activating, previous

    def build_prompt(self, examples, top_logits):

        activating = []
        previous = []
        top_examples = []

        for example in examples:
            delimited_string, act, prev = \
                self.prepare_example(
                    example
                )
            
            activating.append(act)
            previous.append(prev)
            top_examples.append(delimited_string)
            
        top_examples_str = ""
            
        for i, example in enumerate(top_examples):
            top_examples_str += f"Example {i}: {example}\n"

        activating = self.flatten(activating)
        previous = self.flatten(previous)

        prompt = create_prompt(
            l=CONFIG.l, 
            r=CONFIG.r, 
            examples=top_examples, 
            top_logits=top_logits,
            activating=activating,
            previous=previous
        )

        return [
            {
                "role" : "user",
                "content" : prompt, 
            }
        ]