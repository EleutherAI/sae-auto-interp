import asyncio

from ..explainer import Example, Explainer
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
        super().__init__(
            client,
            tokenizer,
            verbose,
            activations,
            cot,
            threshold,
            temperature,
            **generation_kwargs,
        )

    def _build_prompt(self, examples: list[Example]) -> list[dict]:
        highlighted_examples = []

        for i, example in enumerate(examples):
            str_toks = self.tokenizer.batch_decode(example.tokens)
            activations = example.activations.tolist()
            highlighted_examples.append(self._highlight(str_toks, activations))

            if self.activations:
                assert (
                    example.normalized_activations is not None
                ), "Normalized activations are required for activations in explainer"
                normalized_activations = example.normalized_activations.tolist()
                highlighted_examples.append(
                    self._join_activations(
                        str_toks, activations, normalized_activations
                    )
                )

        highlighted_examples = "\n".join(highlighted_examples)

        return build_prompt(
            examples=highlighted_examples,
            activations=self.activations,
            cot=self.cot,
        )

    def call_sync(self, record):
        return asyncio.run(self.__call__(record))
