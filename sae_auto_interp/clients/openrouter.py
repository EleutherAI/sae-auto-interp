from .client import Client
import httpx
from openai import AsyncOpenAI
from ..logger import logger
from asyncio import sleep
import json
import os

PROVIDER = {
    "order": [
        "Together",
        "DeepInfra"
    ]
}

class OpenRouter(Client):
    def __init__(
        self, 
        model: str,
        api_key: str=None,
        base_url="https://openrouter.ai/api/v1/chat/completions"
    ):
        super().__init__(model)

        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }

        self.url = base_url
        self.client = httpx.AsyncClient()

    def postprocess(self, response):
        response_json = response.json()
        return response_json["choices"][0]['message']['content']
    
    async def generate(
        self, 
        prompt: str, 
        raw: bool = False,
        max_retries: int = 3,
        **kwargs
    ) -> str:

        # try:
        #     if kwargs.pop("schema"):
        #         kwargs["response_format"] = {"type" : "json_object"}
        # except KeyError:
        #     pass

        kwargs.pop("schema")

        data = {
            "model": self.model,
            "messages" : prompt,
            # "provider": PROVIDER,
            **kwargs
        }

        for attempt in range(1):

            try:

                response = await self.client.post(
                    url=self.url, 
                    json=data, 
                    headers=self.headers
                )


                if raw:
                    return response
                
                result = self.postprocess(response)
                
                # if "response_format" in kwargs:
                #     return json.loads(result)
                
                return json.loads(result)
            
            except json.JSONDecodeError:
                logger.warning(f"Attempt {attempt + 1}: Invalid JSON response, retrying...")
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {str(e)}, retrying...")
            
            await sleep(1)

        logger.error("All retry attempts failed.")
        raise RuntimeError("Failed to generate text after multiple attempts.")