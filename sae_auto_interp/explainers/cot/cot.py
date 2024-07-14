import re
from typing import List

from .prompts import create_prompt
from ..explainer import (
    Explainer, 
    ExplainerInput, 
)


L = "<<"
R = ">>"

class ChainOfThought(Explainer):
    """
    The Chain of Thought explainer generates an explanation
    using a few shot examples to guide the model in its forming an explanation.
    """

    name = "cot"

    def __init__(
        self,
        client,
        max_tokens=200,
        temperature=0.0,
        threshold=0.3
    ):
        self.client = client

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.threshold = threshold

    async def __call__(
        self,
        explainer_in: ExplainerInput
    ):
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
            max_tokens=self.max_tokens,
            temperature=self.temperature
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

        threshold = example.max_activation * self.threshold

        pos = 0

        while pos < len(example.tokens):
            

            # Check if next token will activate
            if (pos + 1 < len(example.tokens) 
                and example.activations[pos + 1] > threshold
            ):
                delimited_string += example.str_toks[pos]
                previous.append(example.str_toks[pos])
                pos += 1

            # Check if current token activates
            elif example.activations[pos] > threshold:
                delimited_string += L

                # Build activating token chunk and
                # delimited string at the same time
                seq = ""
                while pos < len(example.tokens) and example.activations[pos] > threshold:
                    delimited_string += example.str_toks[pos]
                    seq += example.str_toks[pos]
                    pos += 1
                activating.append(seq)

                delimited_string += R
            
            # Else, keep building the delimited string
            else:
                delimited_string += example.str_toks[pos]
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
            top_examples_str += f"Example {i +1}: {example}\n"

        activating = self.flatten(activating)
        previous = self.flatten(previous)

        prompt = create_prompt(
            l=L, 
            r=R, 
            examples=top_examples_str, 
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