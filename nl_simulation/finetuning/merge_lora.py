from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Load the LoRA model
lora_model = PeftModel.from_pretrained(base_model, "/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/output_quantiles_top5/checkpoint-1000")

# Merge the LoRA weights with the base model
merged_model = lora_model.merge_and_unload()

# Save the merged model
output_dir = "/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/quantiles_top5"
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Merged model saved to {output_dir}")
