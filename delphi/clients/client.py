from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Dict, Any


@dataclass
class Response:
    text: str
    logprobs: List[float] = None
    prompt_logprobs: List[float] = None


class Client(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> Response:
        pass

    @abstractmethod
    async def process_response(self, raw_response: Any) -> Response:
        pass

