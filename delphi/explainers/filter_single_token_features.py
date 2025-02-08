from delphi.logger import logger

from delphi.explainers.default.default import DefaultExplainer
from delphi.explainers.explainer import ExplainerResult
from delphi.features.features import Example, FeatureRecord


class SkipSingleTokenFeaturesExplainer(DefaultExplainer):
    name = "skip_single_token_features"

    def __init__(self, n_examples: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_examples = n_examples

    def _simple_explanation_or_none(
        self, examples: list[Example], n_examples: int
    ) -> tuple[str | None, int | None]:
        """Check if all examples have max activation on the same token.
        Examples is assumed to be sorted by max activation token in descending order."""
        max_activation_tokens = [
            example.max_activation_token.item() for example in examples[:n_examples]
        ]

        # Check if all max activation tokens are the same
        if len(set(max_activation_tokens)) == 1:
            token = max_activation_tokens[0]
            return (
                f"The single token enclosed in angle brackets: <<{self.tokenizer.decode([token])}>>.",
                token,
            )

        print(max_activation_tokens)
        return None, None

    async def __call__(self, record: FeatureRecord):
        # Use templated explanation for single token features
        simple_explanation, token = self._simple_explanation_or_none(
            record.train, self.n_examples
        )
        if simple_explanation is not None:
            print(
                f"Skipping feature {record.feature.feature_index} with simple explanation: {simple_explanation} \
(top {self.n_examples} examples activate on token index {token})"
            )
            return ExplainerResult(record=record, explanation=simple_explanation)

        # Build prompt and get explanation
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
            return ExplainerResult(
                record=record, explanation="Explanation could not be parsed."
            )
