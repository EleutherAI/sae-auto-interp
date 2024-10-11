import asyncio
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..logger import logger
from .client import Client, Response
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

@dataclass
class Top_Logprob:
    token: str
    logprob: float

@dataclass
class Logprobs:
    token: str
    top_logprobs: list[Top_Logprob]

class Offline(Client):
    provider = "offline"

    def __init__(self, model: str, max_memory: float=0.85, prefix_caching: bool=True, batch_size: int=100, max_model_len: int=4096, num_gpus: int=2, enforce_eager: bool=False, lora_path: str=None):
        super().__init__(model)
        self.queue = asyncio.Queue()
        self.task = None
        self.client = LLM(model=model, gpu_memory_utilization=max_memory, enable_prefix_caching=prefix_caching, tensor_parallel_size=num_gpus, max_model_len=max_model_len, enforce_eager=enforce_eager,enable_lora=True)
        self.sampling_params = SamplingParams(max_tokens=500, temperature=0.01)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.batch_size = batch_size
        if lora_path is not None:
            from vllm.lora.request import LoRARequest
            request = LoRARequest("lora_adapter",1,lora_path)
            self.lora_request = request
        else:
            self.lora_request = None
        

    async def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> Response:
        future = asyncio.Future()
        if self.task is None:
            self.task = asyncio.create_task(self._process_batches())
        await self.queue.put((prompt, future, kwargs))
        return await future

    async def process_response(self, raw_response: Any) -> Response:
        logprobs, prompt_logprobs = self._parse_logprobs(raw_response)
        return Response(text=raw_response.outputs[0].text, logprobs=logprobs, prompt_logprobs=prompt_logprobs)

    async def _process_batches(self):
        while True:
            batch = []
            batch_futures = []
            batch_kwargs = []
            start_time = asyncio.get_event_loop().time()
            while len(batch) < self.batch_size:
                try:
                    prompt, future, kwargs = self.queue.get_nowait()
                    batch.append(prompt)
                    batch_futures.append(future)
                    batch_kwargs.append(kwargs)
                except asyncio.QueueEmpty:
                    if batch:
                        break
                    await asyncio.sleep(0.1)
                    continue

                if asyncio.get_event_loop().time() - start_time > 1:
                    break
                
            if not batch:
                continue

            try:
                results = await self._process_func(batch, batch_kwargs)
                for result, future in zip(results, batch_futures):
                    if not future.done():
                        future.set_result(result)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                for future in batch_futures:
                    if not future.done():
                        future.set_exception(e)

    async def _process_func(self, batches: Union[str, List[Dict[str, str]]], kwargs):
        for kwarg in kwargs:
            if "logprobs" in kwarg:
                self.sampling_params.logprobs = kwarg["top_logprobs"]
            if "prompt_logprobs" in kwarg:
                self.sampling_params.prompt_logprobs = kwarg["prompt_logprobs"]
        
        prompts = [self.tokenizer.apply_chat_template(batch, add_generation_prompt=True, tokenize=True) for batch in batches]
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, 
            partial(self.client.generate, prompt_token_ids=prompts, sampling_params=self.sampling_params, use_tqdm=False, lora_request=self.lora_request)
        )
        
        return [await self.process_response(r) for r in response]

    def _parse_logprobs(self, response):
        logprobs = response.outputs[0].logprobs
        prompt_logprobs = response.prompt_logprobs
        if logprobs is None and prompt_logprobs is None:
            return None, None
        
        logprobs_list = None
        if logprobs is not None:
            logprobs_list = []
            for log_prob_dict in logprobs:
                top_logprobs = []
                decoded_token = ""
                for token, logprob in log_prob_dict.items():
                    if logprob.rank == 1:
                        decoded_token = logprob.decoded_token
                    top_logprobs.append(Top_Logprob(token=logprob.decoded_token, logprob=logprob.logprob))
                logprobs_list.append(Logprobs(token=decoded_token, top_logprobs=top_logprobs))
        
        return logprobs_list, prompt_logprobs

    async def close(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.client
        self.client = None
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

