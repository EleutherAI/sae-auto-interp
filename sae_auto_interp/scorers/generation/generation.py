from .prompts import get_gen_scorer_template
from ..scorer import Scorer, ScorerInput
from ...clients import get_client
from pydantic import create_model

def create_str_response_model(n: int):
    fields = {f'example_{i}': (str, ...) for i in range(n)}
    
    ResponseModel = create_model('ResponseModel', **fields)
    
    return ResponseModel

class GenerationScorer(Scorer):
    name = "generation"

    def __init__(
        self,
        client,
        n_examples: int = 10,
        temperature: float = 0.5,
        max_tokens: int = 1000,
    ):
        self.client = client
        self.n_examples = n_examples

        self.temperature = temperature
        self.max_tokens = max_tokens

    async def __call__(
        self, 
        scorer_in: ScorerInput,
    ):
        
        prompt = get_gen_scorer_template(
            scorer_in.explanation,
            self.n_examples
        )

        schema = create_str_response_model(
            self.n_examples
        )

        generation_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # Generate responses
        examples = await self.client.generate(
            prompt,
            schema=schema.model_json_schema(),
            **generation_kwargs
        )

        return list(examples.values())