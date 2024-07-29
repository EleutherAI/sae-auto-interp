from .client import Client
import httpx
from openai import AsyncOpenAI
from ..logger import logger
from asyncio import sleep
import json
import os
import re
# Preferred provider routing arguments. 
# Change depending on what model you'd like to use.
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
        max_retries: int = 1,
        **kwargs
    ) -> str:

        schema = kwargs.pop("schema", None)
        data = {
            "model": self.model,
            "messages" : prompt,
            # "provider": PROVIDER,
            **kwargs
        }
        for attempt in range(max_retries):

            try:
                response = await self.client.post(
                    url=self.url, 
                    json=data, 
                    headers=self.headers
                )

                if raw:
                    return response.json()
                
                result = self.postprocess(response)
                if schema is not None:
                    # Patern match the response to a valid json
                    pattern = r'\{[^{}]*\}'
                    matches = re.findall(pattern, result)
                    if len(matches) > 0:
                        processed_response = matches[0]
                        processed_response = json.loads(processed_response)
                    else:
                        logger.warning("Invalid response structure.")
                        raise json.JSONDecodeError("Invalid response structure.", processed_response, 0)

                return processed_response
            
            except json.JSONDecodeError:
                logger.warning(f"Attempt {attempt + 1}: Invalid JSON response, retrying...")
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {str(e)}, retrying...")
            
                await sleep(1)

        logger.error("All retry attempts failed.")
        raise RuntimeError("Failed to generate text after multiple attempts.")