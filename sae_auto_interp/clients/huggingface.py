
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..logger import logger
from .client import Client

class HuggingFace(Client):
    provider = "huggingface"

    def __init__(self, model: str):
        super().__init__(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        self.model = torch.compile(self.model)

    async def generate(
        self, 
        prompt: str, 
        **kwargs
    ) -> str:
        """
        """
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        input_ids = self.tokenizer.apply_chat_template(prompt, tokenize=True, return_tensors="pt").to("cuda")
        #print(input_ids)
        attention_mask = input_ids != self.tokenizer.pad_token_id
        attention_mask = attention_mask.to("cuda")
        response = self.model.generate(
            input_ids,
            max_new_tokens=200,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        start_of_response = len(input_ids[0])
        generated_ids = response.sequences[0][start_of_response:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response 
