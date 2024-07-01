from .client import Client
import httpx
from transformers import AutoTokenizer

class Local(Client):
    def __init__(self, model: str):
        super().__init__(model)

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.client =httpx.AsyncClient(
            base_url="http://127.0.0.1:8000",
            timeout=None
        )

    def generate(self):
        pass

    def postprocess(self, prompt, response):
        response_text = response.json()["text"][0]
        return response_text[len(prompt) : ]


    async def async_generate(self, prompt: str, tokenize=True, raw=False, echo=False, **kwargs) -> str:
        if tokenize:
            prompt = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )

        data = {
            "prompt" : prompt,
            **kwargs
        }
        
        response = await self.client.post("/generate", json=data)

        if raw:
            return response
        
        if echo:
            return prompt, self.postprocess(prompt, response)
        
        return self.postprocess(prompt, response)