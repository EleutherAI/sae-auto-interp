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

    def postprocess(self, prompt, response):
        response_text = response.json()["text"][0]
        return response_text[len(prompt) : ]

    async def generate(
        self, 
        prompt: str, 
        tokenize: bool =True, 
        raw: bool =False, 
        echo: bool =False, 
        **kwargs
    ) -> str:
        """
        Wrapper method for Outlines/vLLM post requests.
        """

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