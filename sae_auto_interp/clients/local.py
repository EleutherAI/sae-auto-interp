from .client import Client
from openai import AsyncOpenAI, OpenAI

# NOTE: Currently only uses the async client
class Local(Client):
    def __init__(self, model: str):
        super().__init__(model)

        self.client = AsyncOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="n/a"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        return self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature = kwargs.get("temperature", 0.0),
            max_tokens = kwargs.get("max_tokens", 1000),
        ).choices[0].message.content

    async def async_generate(self, prompt: str, **kwargs) -> str:
        value = await self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature = kwargs.get("temperature", 0.0),
            max_tokens = kwargs.get("max_tokens", 1000),
        )
        return value.choices[0].message.content
