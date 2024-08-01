from abc import ABC, abstractmethod

from pydantic import create_model


class Client(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
<<<<<<< HEAD:src/sae_auto_interp/clients/client.py
    async def generate(
        self, 
        prompt: str, 
        **kwargs
    ):
=======
    async def generate(self, prompt: str, **kwargs) -> str:
>>>>>>> ca973f5a5f4c1feaafaf0dae94c9a3f068104774:sae_auto_interp/clients/client.py
        raise NotImplementedError


def create_response_model(n: int, type: type = int):
    fields = {f"example_{i}": (type, ...) for i in range(n)}

    ResponseModel = create_model("ResponseModel", **fields)

    return ResponseModel
