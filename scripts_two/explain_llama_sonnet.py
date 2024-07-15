import os
os.environ["CONFIG_PATH"] = "/mnt/ssd-1/gpaulo/SAE-Zoology/sae_auto_interp/configs/gpaulo_llama_256.yaml"
import asyncio
from nnsight import LanguageModel 
import torch
from sae_auto_interp.explainers import SimpleExplainer,ChainOfThought, ExplainerInput
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import execute_model, load_tokenized_data
from sae_auto_interp.features import FeatureRecord
import random
import argparse



argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="0,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]

openrouter_key = os.environ["OPENROUTER_KEY"]

# Load model 
model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="cpu", dispatch=True,torch_dtype =torch.bfloat16)
print("Model loaded")
# Load tokenized data
n_tokens = 10_400_000
dataset_repo =  "kh4dien/fineweb-100m-sample"
batch_len = 256
tokens  = load_tokenized_data(model.tokenizer,n_tokens=n_tokens, dataset_repo=dataset_repo, batch_len=batch_len,dataset_split="train")
print("Tokenized data loaded")
# Load features to explain

# Raw features contains locations
raw_features_path = "raw_features_llama"

explainer_inputs=[]
seed = 22
random.seed(seed)
        
for layer in layers:
    records = FeatureRecord.from_tensor(
        tokens,
        layer_index=layer,
        selected_features=torch.arange(0,50),
        raw_dir= raw_features_path,
        processed_dir="processed_features_llama",
        max_examples=100000
    )

    for record in records:
        all_examples = record.examples
        if len(all_examples) < 500:
            continue
        top500 = all_examples[:500]

        examples = random.sample(top500, 20)+random.sample(all_examples[:-5], 15)+all_examples[-5:]

        for example in examples:
            example.decode(model.tokenizer)
            
        
        explainer_inputs.append(
            ExplainerInput(
                train_examples=examples,
                record=record
            )
        )
        


#client = get_client("local", "meta-llama/Meta-Llama-3-8B-Instruct", base_url="http://127.0.0.1:8002")
client = get_client("openrouter", "anthropic/claude-3.5-sonnet", api_key=openrouter_key)

explainer = SimpleExplainer(client)
print("Running 1")
explainer_out_dir = "saved_explanations/llama_sonnet/"
asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
    )
)
