import re

import torch

from ..explainer import Explainer, ExplainerResult
from .prompt_builder import build_prompt


class SimpleExplainer(Explainer):
    name = "Simple"

    def __init__(
        self,
        client,
        tokenizer,
        verbose: bool = False,
        cot: bool = False,
        logits: bool = False,
        activations: bool = False,
        threshold: float = 0.6,
        **generation_kwargs,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.verbose = verbose

        self.cot = cot
        self.logits = logits
        self.activations = activations

        self.threshold = threshold
        self.generation_kwargs = generation_kwargs

    def normalize_examples(self, record, train):
        max_activation = record.examples[0].max_activation

        for example in train:
            example.normalized_activations = torch.floor(
                10 * example.activations / max_activation
            )

    async def __call__(self, record):
        if self.activations:
            self.normalize_examples(record, record.train)

        if self.logits:
            messages = self._build_prompt(record.train, record.top_logits)
        else:
            messages = self._build_prompt(record.train, None)

        response = await self.client.generate(messages, **self.generation_kwargs)

        explanation = self.parse_explanation(response)

        if self.verbose:
            return (
                messages[-1]["content"],
                response,
                ExplainerResult(record=record, explanation=explanation),
            )

        return ExplainerResult(record=record, explanation=explanation)

    def parse_explanation(self, text: str) -> str:
        match = re.search(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL)

        return match.group(1).strip() if match else "Explanation could not be parsed."

    def _highlight(self, index, example):
        result = f"Example {index}: "

        threshold = example.max_activation * self.threshold
        str_toks = self.tokenizer.batch_decode(example.tokens)
        example.str_toks = str_toks
        activations = example.activations

        def check(i):
            return activations[i] > threshold

        i = 0
        while i < len(str_toks):
            if check(i):
                result += "<<"

                while i < len(str_toks) and check(i):
                    result += str_toks[i]
                    i += 1
                result += ">>"
            else:
                result += str_toks[i]
                i += 1

        return "".join(result)

    def _join_activations(self, example):
        activations = []

        threshold = example.max_activation * 0.7
        for i, normalized in enumerate(example.normalized_activations):
            if example.activations[i] > threshold:
                activations.append((example.str_toks[i], int(normalized)))

        acts = ", ".join(f'("{item[0]}" : {item[1]})' for item in activations)

        return "Activations: " + acts

    def _build_prompt(self, examples, top_logits):
        highlighted_examples = []

        for i, example in enumerate(examples):
            highlighted_examples.append(self._highlight(i + 1, example))

            if self.activations:
                highlighted_examples.append(self._join_activations(example))

        highlighted_examples = "\n".join(highlighted_examples)

        return build_prompt(
            examples=highlighted_examples,
            cot=self.cot,
            activations=self.activations,
            top_logits=top_logits,
        )
