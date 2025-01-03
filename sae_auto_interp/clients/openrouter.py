import json
from asyncio import sleep
from typing import Any, Dict, List, Union

import httpx

from ..logger import logger
from .client import Client, Response


class OpenRouter(Client):
    def __init__(
        self,
        model: str,
        api_key: str = None,
        base_url="https://openrouter.ai/api/v1/chat/completions",
    ):
        super().__init__(model)
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.url = base_url
        self.client = httpx.AsyncClient()

    async def generate(self, prompt: Union[str, List[Dict[str, str]]], max_retries: int = 2, **kwargs) -> Response:
        kwargs.pop("schema", None)
        data = {
            "model": self.model,
            "messages": prompt,
            **kwargs,
        }

        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    url=self.url, json=data, headers=self.headers
                )
                return await self.process_response(response)
            except json.JSONDecodeError:
                logger.warning(f"Attempt {attempt + 1}: Invalid JSON response, retrying...")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {str(e)}, retrying...")
            await sleep(1)

        logger.error("All retry attempts failed.")
        raise RuntimeError("Failed to generate text after multiple attempts.")

    async def process_response(self, raw_response: Any) -> Response:
        response_json = raw_response.json()
        text = response_json["choices"][0]["message"]["content"]
        return Response(text=text)
