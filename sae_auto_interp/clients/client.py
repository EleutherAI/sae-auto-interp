from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class Response:
    text: str
    logprobs: List[float]
    prompt_logprobs: List[float]


class Client(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        **kwargs
    ):
        raise NotImplementedError

