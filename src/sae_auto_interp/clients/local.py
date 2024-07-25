from .client import Client
from ..logger import logger
from asyncio import sleep
import json

from openai import AsyncOpenAI

import logging

# Define a custom log level
CUSTOM_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(CUSTOM_LEVEL, "CUSTOM")

# Create a method for the custom level
def custom(self, message, *args, **kwargs):
    if self.isEnabledFor(CUSTOM_LEVEL):
        self._log(CUSTOM_LEVEL, message, args, **kwargs)

# Add the custom level method to the Logger class
logging.Logger.custom = custom

# Configure the logger
logging.basicConfig(filename='token_stats/vllm.log', level=CUSTOM_LEVEL,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Get the logger
log = logging.getLogger(__name__)

class Local(Client):
    provider = "vllm"

    def __init__(self,
        model: str, 
        base_url="http://localhost:8000/v1"
    ):
        super().__init__(model)
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=None
        )
        self.model = model

    async def generate(
        self, 
        prompt: str, 
        raw: bool = False,
        max_retries: int = 2,
        **kwargs
    ) -> str:
        """
        Wrapper method for vLLM post requests.
        """

        for attempt in range(max_retries):

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    **kwargs
                )

                log.custom(response.usage)

                return response if raw else self.postprocess(response)

            except json.JSONDecodeError:
                logger.warning(f"Attempt {attempt + 1}: Invalid JSON response, retrying...")
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {str(e)}, retrying...")
            
            await sleep(1)

        logger.error("All retry attempts failed.")
        raise RuntimeError("Failed to generate text after multiple attempts.")
    
    def postprocess(self, response: dict) -> str:
        """
        Postprocess the response from the API.
        """
        return response.choices[0].message.content