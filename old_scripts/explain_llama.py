import asyncio
from nnsight import LanguageModel 
from tqdm import tqdm
import torch
from sae_auto_interp.explainers import SimpleExplainer, ExplainerInput, ChainOfThought
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import execute_model, load_tokenized_data, get_samples
from sae_auto_interp.features import FeatureRecord
import random
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]

# Load model 
model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="cpu", dispatch=True,torch_dtype =torch.bfloat16)
print("Model loaded")
# Load tokenized data
tokens = load_tokenized_data(model.tokenizer)
print("Tokenized data loaded")
# Load features to explain
samples = get_samples(N_LAYERS=32,N_FEATURES=131072,N_SAMPLES=1000)

# Raw features contains locations
raw_features_path = "raw_features_llama"
processed_features_path = "processed_features_llama"
explainer_inputs=[]
seed = 22
random.seed(seed)
        
for layer in layers:
    records = FeatureRecord.from_tensor(
        tokens,
        tokenizer=model.tokenizer,
        layer_index=layer,
        selected_features=samples[layer],
        processed_dir=processed_features_path,
        raw_dir= raw_features_path,
        max_examples=2000
    )

    for record in records:
        all_examples = record.examples
        if len(all_examples) < 100:
            continue
        top100 = all_examples[:100]

        random_10_all_10 = all_examples[:10]
        
        explainer_inputs.append(
            ExplainerInput(
                train_examples=random_10_all_10,
                record=record
            )
        )


client = get_client("local", "meta-llama/Meta-Llama-3-8B-Instruct", base_url="http://127.0.0.1:8001")

explainer = ChainOfThought(client)
print("Running explainer")
explainer_out_dir = "saved_explanations/llama_cot/"
asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
    )
)

