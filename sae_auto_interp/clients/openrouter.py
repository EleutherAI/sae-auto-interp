from .client import Client
import httpx
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from ..logger import logger
from asyncio import sleep
import json
import os

class OpenRouter(Client):
    def __init__(
        self, 
        model: str,
        api_key: str=None,
        base_url="https://openrouter.ai/api/v1"
    ):
        super().__init__(model)

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def postprocess(self, response):
        return response.choices[0].message.content

    async def generate(
        self, 
        prompt: str, 
        raw: bool = False,
        max_retries: int = 3,
        **kwargs
    ) -> str:

        try:
            if kwargs.pop("schema"):
                kwargs["response_format"] = {"type" : "json_object"}
        except KeyError:
            pass

        for attempt in range(max_retries):

            try:
                response = await self.client.chat.completions.create(
                    model = self.model,
                    messages = prompt,
                    **kwargs
                )

                if raw:
                    return response
                
                result = self.postprocess(response)
                if "response_format" in kwargs:
                    return json.loads(result)
                
                return result
            
            except json.JSONDecodeError:
                logger.warning(f"Attempt {attempt + 1}: Invalid JSON response, retrying...")
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {str(e)}, retrying...")
            
            await sleep(1)

        logger.error("All retry attempts failed.")
        raise RuntimeError("Failed to generate text after multiple attempts.")