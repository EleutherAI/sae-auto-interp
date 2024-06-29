from .local import Local
import os

openrouter_key = os.environ.get("OPENAI_API_KEY")

def get_client(provider: str, model: str):
    if provider == "local":
        return Local(model)
    return None