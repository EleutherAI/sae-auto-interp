from abc import ABC, abstractmethod

from pydantic import create_model


class Client(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


def create_response_model(n: int, type: type = int):
    fields = {f"example_{i}": (type, ...) for i in range(n)}

    ResponseModel = create_model("ResponseModel", **fields)

    return ResponseModel
