import asyncio
import re

from sae_auto_interp.explainers.explainer import Explainer, ExplainerResult
from sae_auto_interp.explainers.prompt_builder import build_prompt

from ...logger import logger


class OutputExplainer(Explainer):
    name = "output"

    def __init__(
        self,
        explanation_client,
        prepared_examples,
        tokenizer,
        verbose: bool = False,
        logit_lens: bool = False,
        **generation_kwargs,
    ):
        self.client = explanation_client
        self.prepared_examples = prepared_examples
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.logit_lens = logit_lens
        self.generation_kwargs = generation_kwargs


    async def __call__(self, record):
       
        examples = self.prepared_examples
        assert len(examples) > 0, "Prepared examples first"




        messages = self._build_prompt(record.train)
        
        response = await self.client.generate(messages, **self.generation_kwargs)

        try:
            explanation = self.parse_explanation(response.text)
            if self.verbose:
                return (
                    messages[-1]["content"],
                    response,
                    ExplainerResult(record=record, explanation=explanation),
                )

            return ExplainerResult(record=record, explanation=explanation)
        except Exception as e:
            logger.error(f"Explanation parsing failed: {e}")
            return ExplainerResult(record=record, explanation="Explanation could not be parsed.")




    def parse_explanation(self, text: str) -> str:
        try:
            match = re.search(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL)
            return match.group(1).strip() if match else "Explanation could not be parsed."
        except Exception as e:
            logger.error(f"Explanation parsing regex failed: {e}")
            raise
    
    



    def _join_activations(self, example):
        activations = []

        for i, activation in enumerate(example.activations):
            if activation > example.max_activation * self.threshold:
                activations.append((example.str_toks[i], int(example.normalized_activations[i])))

        acts = ", ".join(f'("{item[0]}" : {item[1]})' for item in activations)

        return "Activations: " + acts

    def _build_prompt(self, examples):
        highlighted_examples = []

        for i, example in enumerate(examples):
            highlighted_examples.append(self._highlight(i + 1, example))

            if self.activations:
                highlighted_examples.append(self._join_activations(example))

        highlighted_examples = "\n".join(highlighted_examples)

        return build_prompt(
            examples=highlighted_examples,
            activations=self.activations,
            cot=self.cot,
        )

    def call_sync(self, record):
        return asyncio.run(self.__call__(record))
