from transformers import AutoTokenizer

# from .openai import OpenAI
# from .replicate import Replicate
# from .groq import Groq
# from .openrouter import OpenRouter
from .local import Local
import os

openrouter_key = os.environ.get("OPENAI_API_KEY")

def get_client(provider: str, model: str):
    # if provider is None or api_key is None:
    #     return None 

    # if provider == "openai":
    #     model = client_cfg.openai_model
    #     return OpenAI(model, api_key)
    
    # if provider == "replicate":
    #     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    #     model = client_cfg.replicate_model
    #     return Replicate(model, api_key, tokenizer)

    # if provider == "groq":
    #     model = client_cfg.groq_model
    #     return Groq(model, api_key)
    
    # if provider == "openrouter": 
    #     return OpenRouter(model, api_key=openrouter_key)

    if provider == "local":
        return Local(model)
