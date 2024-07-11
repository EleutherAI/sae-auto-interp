from .local import Local
from .openrouter import OpenRouter
from .outlines import Outlines

def get_client(provider: str, model: str, **kwargs):
    if provider == "local":
        return Local(model=model, **kwargs)
    if provider == "openrouter":
        return OpenRouter(model=model, **kwargs)
    if provider == "outlines":
        return Outlines(model=model, **kwargs)

    return None