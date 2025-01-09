import json
from asyncio import sleep
from typing import Any, Dict, List, Union
import traceback

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

    async def generate(
        self, prompt: str, raw: bool = False, max_retries: int = 1, **kwargs
    ) -> str:
        kwargs.pop("schema", None)
        max_tokens = kwargs.pop("max_tokens", 500)
        temperature = kwargs.pop("temperature", 1.0)
        data = {
            "model": self.model,
            "messages": prompt,
            # "provider": PROVIDER,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    url=self.url, json=data, headers=self.headers
                )
                if raw:
                    return response.json()
                result = self.process_response(response)

                return result

            except json.JSONDecodeError:
                traceback.print_exc()
                logger.warning(
                    f"Attempt {attempt + 1}: Invalid JSON response, retrying..."
                )

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {str(e)}, retrying...")

            await sleep(1)

        logger.error("All retry attempts failed.")
        raise RuntimeError("Failed to generate text after multiple attempts.")

    def process_response(self, response):
        response_json = response.json()
        msg = response_json["choices"][0]["message"]["content"]
        return Response(msg)
