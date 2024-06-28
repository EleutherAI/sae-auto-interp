from .client import Client
from typing import List

class Local(Client):
    def __init__(self, model: str):
        super().__init__(model)
        config = {
        "backend": "vllm",
        "model": "casperhansen/llama-3-70b-instruct-awq",
        "quantization": "awq"
        }
        self.initialize_backend(config)

    def initialize_backend(self,config:dict[str,str]):
        backend = config["backend"]
        self.backend = backend
        self.model = config["model"]
        if backend=="vllm":
            from vllm import LLM
            quantization = config["quantization"]
            self.llm = LLM(model=self.model,quantization=quantization,load_format="auto",enable_prefix_caching=True,gpu_memory_utilization=0.9)
        elif backend=="llama_cpp":
            from llama_cpp import Llama
            filename = config["filename"]
            ##TODO: Context should be model dependent
            self.llm = Llama.from_pretrained(
                repo_id=self.model,
                filename=filename,
                n_gpu_layers=-1,
                n_ctx=8192,
                verbose=False
            )

    def generate(self, prompt: str, generation_args: dict) -> str:
        if self.backend=="vllm":
            from vllm import SamplingParams
            sampling = SamplingParams(max_tokens=generation_args.get("max_tokens", 100))
            return self.llm.generate(prompt,sampling)[0].outputs[0].text
        elif self.backend=="llama_cpp":
            return self.llm.create_chat_completion(prompt,stop=".",max_tokens=generation_args.get("max_tokens", 100))["choices"][0]["message"]["content"]
        else:
            raise NotImplementedError("Backend not implemented")
    
    def generate_batch(self, prompts: List[str], generation_args: dict) -> List[str]:
        if self.backend=="vllm":
            from vllm import SamplingParams
            sampling = SamplingParams(max_tokens=generation_args.get("max_tokens", 100))
            answers = self.llm.generate(prompts,sampling)
            return [answer.outputs[0].text for answer in answers]
        elif self.backend=="llama_cpp":
            print("Batching not supported for llama_cpp, falling back to single generation")
            return [self.llm.create_chat_completion(prompt,stop=".",max_tokens=generation_args.get("max_tokens", 100))["choices"][0]["message"]["content"] for prompt in prompts]
        else:
            raise NotImplementedError("Backend not implemented")
