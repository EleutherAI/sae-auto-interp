from .local import Local

def get_client(provider: str, model: str):
    if provider == "local":
        return Local(model)
    return None