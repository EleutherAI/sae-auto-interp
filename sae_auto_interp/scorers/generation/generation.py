from ...clients import Client, create_response_model
from ..scorer import Scorer, ScorerResult
from .prompts import get_gen_scorer_template


class GenerationScorer(Scorer):
    name = "generation"

    def __init__(self, client: Client, n_examples: int = 10, **generation_kwargs):
        self.client = client
        self.n_examples = n_examples

        self.generation_kwargs = generation_kwargs

    async def __call__(
        self,
        record,
    ):
        prompt = get_gen_scorer_template(record.explanation, self.n_examples)

        schema = create_response_model(self.n_examples, type=str)

        # Generate responses
        examples = await self.client.generate(
            prompt, schema=schema.model_json_schema(), **self.generation_kwargs
        )

        return ScorerResult(record=record, score=list(examples.values()))
