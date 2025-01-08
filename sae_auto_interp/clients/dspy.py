import asyncio
import traceback
from types import SimpleNamespace
from typing import Any

import dspy
import litellm

from ..logger import logger
from .client import Client, Response


class DSPy(Client):
    """
    DSPy Client.
    """

    provider = "dspy"
    
    def __init__(self, dspy_client: dspy.LM):
        super().__init__(dspy_client.model)
        self.client = dspy_client
        self.client.num_retries = 0

    async def generate(
        self, 
        prompt: str,
        max_retries: int = 4,
        sleep_time: float = 90.0,
        **kwargs
    ):
        kwargs = dict(
            max_tokens=500,
            temperature=1.0
        ) | kwargs
        logger.debug(f"DSPy prompt input: {prompt}")
        for i in range(max_retries):
            try:
                if isinstance(prompt, list) and isinstance(prompt[0], dict):
                    response = self.client(messages=prompt, **kwargs)
                else:
                    response = self.client(prompt, **kwargs)
                logger.debug(f"DSPy prompt: {prompt}")
                logger.debug(f"DSPy gen: {response}")
                return SimpleNamespace(text=response[0])
            except litellm.RateLimitError:
                traceback.print_exc()
                if i < max_retries - 1:
                    logger.warning(f"Attempt {i+1} failed, retrying...")
                    await asyncio.sleep(sleep_time)

        logger.error("All retry attempts failed.")
        raise RuntimeError("Failed to generate text after multiple attempts.")

    async def process_response(self, raw_response: Any) -> Response:
        return raw_response
