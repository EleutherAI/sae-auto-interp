from .local import Local

def get_client(provider: str, model: str, **kwargs):
    if provider == "local":
        return Local(model, **kwargs)
    return None