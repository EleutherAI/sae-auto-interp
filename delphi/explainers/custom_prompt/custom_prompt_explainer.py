from typing import Protocol

from ..default.default import DefaultExplainer


class PromptBuilder(Protocol):
    def __call__(self, highlighted_examples: str) -> list[dict[str, str]]: ...


class CustomPromptExplainer(DefaultExplainer):
    name = "custom_prompt"

    def __init__(
        self,
        client,
        tokenizer,
        prompt_builder: PromptBuilder,
        activations: bool = False,
        verbose: bool = False,
        threshold: float = 0.6,
        **generation_kwargs,
    ):
        """
        Explainer that generates explanations using a custom prompt builder.
        
        Args:
            client: The language model client
            tokenizer: Tokenizer for processing input text
            build_prompt: A function that constructs the prompt messages for the language model.
                Args:
                    highlighted_examples (str): Text with the activating words highlighted using <<word>> and 
                    optionally their activation values.
                Returns:
                    list[dict[str, str]]: List of message dicts in the format:
                    [
                        {"role": "assistant"|"user", "content": str},
                        ...
                    ]
                    to be passed to the language model for explanation generation. The content of the final message is expected
                    to contain the highlighted examples.
            activations (bool): Whether to include activation values in the input to the prompt builder.
            verbose: Whether to return additional debug information.
            threshold: Activation threshold for highlighting tokens.
            **generation_kwargs: Additional kwargs for the language model.
        """
        super().__init__(
            client=client,
            tokenizer=tokenizer,
            verbose=verbose,
            activations=activations,
            threshold=threshold,
            **generation_kwargs
        )
        self.prompt_builder = prompt_builder

    def _build_prompt(self, examples):
        highlighted_examples = []

        for i, example in enumerate(examples):
            highlighted_examples.append(self._highlight(i + 1, example))

            if self.activations:
                highlighted_examples.append(self._join_activations(example))

        highlighted_examples = "\n".join(highlighted_examples)

        return self.prompt_builder(highlighted_examples)
    