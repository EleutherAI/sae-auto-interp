from ...clients import Client, create_response_model
from ..scorer import Scorer, ScorerResult
from .prompts import get_gen_scorer_template
import re
import json

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

        #schema = create_response_model(self.n_examples, type=str)

        # Generate responses
        examples = await self.client.generate(
            prompt, **self.generation_kwargs
        )

        try:
            match = re.search(r"\{.*\}", examples, re.DOTALL)
            array = json.loads(match.group(0))
            
        except:
            return ScorerResult(record=record, score=list(len(examples)*[" "]))

        return ScorerResult(record=record, score=list(array.values()))
