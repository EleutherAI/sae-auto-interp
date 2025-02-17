import asyncio
import re

from ...logger import logger
from ..explainer import Explainer, ExplainerResult
from .prompt_builder import build_prompt


class DefaultExplainer(Explainer):
    name = "default"

    def __init__(
        self,
        client,
        tokenizer,
        verbose: bool = False,
        activations: bool = False,
        cot: bool = False,
        threshold: float = 0.6,
        temperature: float = 0.0,
        **generation_kwargs,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.verbose = verbose

        self.activations = activations
        self.cot = cot
        self.threshold = threshold
        self.temperature = temperature
        self.generation_kwargs = generation_kwargs


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
