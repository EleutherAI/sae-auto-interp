from .client import Client
import httpx
from transformers import AutoTokenizer
from ..logger import logger
from asyncio import sleep
import json

class Outlines(Client):
    provider = "outlines"

    def __init__(self,
        model: str, 
        base_url="http://127.0.0.1:8000"
    ):
        super().__init__(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=None
        )

    def postprocess(self, prompt, response):
        """
        Convert the response to a JSON object then 
        remove the prompt from the response text.
        """

        response_json = response.json()
        if "text" in response_json and len(response_json["text"]) > 0:
            response_text = response_json["text"][0]
            return response_text[len(prompt):]
        else:
            logger.error("Invalid response structure.")
            return ""

    async def generate(
        self, 
        prompt: str, 
        tokenize: bool = True, 
        raw: bool = False,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Wrapper for async requests to the Outlines vLLM inference engine.
        """

        if tokenize:
            prompt = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )

        data = {
            "prompt": prompt,
            **kwargs
        }
        
        for attempt in range(max_retries):

            try:
                #print(data)
                response = await self.client.post("/generate", json=data)
                
                response.raise_for_status()  
                if raw:
                    return response.json()
                
                response = self.postprocess(prompt, response)
                if kwargs.get("schema") is not None:
                    return json.loads(response)
                
                return response
            
            except json.JSONDecodeError:
                logger.warning(f"Attempt {attempt + 1}: Invalid JSON response, retrying...")
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {str(e)}, retrying...")
            
            await sleep(1)

        logger.error("All retry attempts failed.")
        raise RuntimeError("Failed to generate text after multiple attempts.")