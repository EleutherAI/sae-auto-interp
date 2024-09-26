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

    def __init__(self, model: str, max_memory: float=0.85,prefix_caching:bool=True,batch_size:int=100,max_model_len:int=4096,num_gpus:int=2,enforce_eager:bool=False):
        super().__init__(model)
        self.model = model  
        self.queue = asyncio.Queue()
        self.task = None
        self.client = LLM(model=model, gpu_memory_utilization=max_memory, enable_prefix_caching=prefix_caching, tensor_parallel_size=num_gpus, max_model_len=max_model_len,enforce_eager=enforce_eager)
        self.sampling_params = SamplingParams(max_tokens=500, temperature=0.7)
        self.tokenizer= AutoTokenizer.from_pretrained(model)
        self.batch_size=batch_size
        

    async def process_func(self, batches: Union[str, List[Dict[str, str]]], kwargs):
        """
        Process a single request.
        """

        # This is actually stupid
        for kwarg in kwargs:
            if "logprobs" in kwarg:
                self.sampling_params.logprobs = kwarg["top_logprobs"]
            if "prompt_logprobs" in kwarg:
                self.sampling_params.prompt_logprobs = kwarg["prompt_logprobs"]
        loop = asyncio.get_running_loop()
        prompts=[]
        for batch in batches:
            prompt = self.tokenizer.apply_chat_template(batch, add_generation_prompt=True, tokenize=True)
            prompts.append(prompt)
        response = await loop.run_in_executor(
            None, 
            partial(self.client.generate, prompt_token_ids=prompts, sampling_params=self.sampling_params, use_tqdm=False)
        )
        
        new_response = []   
        for r in response:
            logprobs,prompt_logprobs=self._parse_logprobs(r)
            new_response.append(Response(text=r.outputs[0].text, logprobs=logprobs, prompt_logprobs=prompt_logprobs))
        return new_response

    async def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        """
        Enqueue a request and wait for the result.
        """
        future = asyncio.Future()
        if self.task is None:
            self.task = asyncio.create_task(self._process_batches())
        await self.queue.put((prompt, future, kwargs))
        #print(f"Current queue size: {self.queue.qsize()} prompts")
        return await future

    def _parse_logprobs(self,response):
        logprobs=response.outputs[0].logprobs
        prompt_logprobs=response.prompt_logprobs
        if logprobs is None and prompt_logprobs is None:
            return None,None
        logprobs_list=None
        if logprobs is not None:
            logprobs_list=[]
            for i in range(len(logprobs)):
                log_prob_dict = logprobs[i]
                top_logprobs = []
                decoded_token = ""
                for token, logprob in log_prob_dict.items():
                    if logprob.rank==1:
                        decoded_token = logprob.decoded_token
                        top_logprobs.append(Top_Logprob(token=decoded_token, logprob=logprob.logprob))
                    else:
                        top_logprobs.append(Top_Logprob(token=logprob.decoded_token, logprob=logprob.logprob))
                logprobs_list.append(Logprobs(token=decoded_token, top_logprobs=top_logprobs))
            
        return logprobs_list,prompt_logprobs
            
        

    async def _process_batches(self):
        """
        Continuously process batches of requests.
        """
        batch_count = 0
        while True:
            batch = []
            batch_futures = []
            batch_kwargs = []
            # Collect a batch of requests
            start_time = asyncio.get_event_loop().time()
            while len(batch) < self.batch_size:
                try:
                    prompt, future, kwargs = self.queue.get_nowait()
                    batch.append(prompt)
                    batch_futures.append(future)
                    batch_kwargs.append(kwargs)
                except asyncio.QueueEmpty:
                    if batch:  # If we have any items, process them
                        break
                    await asyncio.sleep(0.1)  # Short sleep if queue is empty
                    continue

                if asyncio.get_event_loop().time() - start_time > 1:  # Time-based batch cutoff
                    break
                
            if not batch:
                continue
            # Process the batch
            try:
                results = await self.process_func(batch, batch_kwargs)
                batch_count += 1
                #print(f"Batch {batch_count} finished processing. {len(results)} prompts processed.")
                for result, future in zip(results, batch_futures):
                    if not future.done():
                        future.set_result(result)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                for future in batch_futures:
                    if not future.done():
                        future.set_exception(e)


    async def close(self):
        """
        Clean up resources when the client is no longer needed.
        """
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
        
        
        