import json
import asyncio

from openai import AsyncOpenAI
from transformers import AutoTokenizer
from ..logger import logger
from .client import Client, Response


class Local(Client):
    provider = "vllm"

    def __init__(self, model: str, base_url="http://localhost:8000/v1"):
        super().__init__(model)
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=None)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    async def generate(self, prompt: Union[str, List[Dict[str, str]]], use_legacy_api: bool = False, max_retries: int = 2, **kwargs) -> Response:
        try:
            for attempt in range(max_retries):
                try:
                    if use_legacy_api:
                        response = await self.client.completions.create(
                            model=self.model,
                            prompt=prompt,
                            **kwargs
                        )
                    else:
                        response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=prompt,
                            **kwargs
                        )
                    if response is None:
                        raise Exception("Response is None")
                    return await self.process_response(response)
                except json.JSONDecodeError as e:
                    logger.warning(f"Attempt {attempt + 1}: Invalid JSON response, retrying... {e}")
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}: {str(e)}, retrying...")
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"All retry attempts failed. Most recent error: {e}")
            raise

    async def process_response(self, raw_response: Any) -> Response:
        return Response(
            text=raw_response.choices[0].message.content,
            logprobs=raw_response.choices[0].logprobs,
            prompt_logprobs=raw_response.choices[0].prompt_logprobs
        )



