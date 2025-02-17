import asyncio
import re

from delphi.explainers.default.prompt_builder import build_single_token_prompt
from delphi.explainers.explainer import Explainer, ExplainerResult
from delphi.logger import logger


class SingleTokenExplainer(Explainer):
    name = "single_token"

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

        return build_single_token_prompt(
            examples=highlighted_examples,
        )

    def call_sync(self, record):
        return asyncio.run(self.__call__(record))
