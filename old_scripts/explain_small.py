import asyncio
from nnsight import LanguageModel 
from tqdm import tqdm
import torch
from sae_auto_interp.explainers import SimpleExplainer, ExplainerInput
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import execute_model, load_tokenized_data, get_samples
from sae_auto_interp.features import FeatureRecord
import random
import argparse
from sae_auto_interp import cache_config as CONFIG

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]

# Load model 
model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="cpu", dispatch=True,torch_dtype =torch.bfloat16)
print("Model loaded")
# Load tokenized data
CONFIG.n_tokens = 10_400_000
CONFIG.dataset_repo =  "kh4dien/fineweb-100m-sample"
CONFIG.batch_len = 256

tokens = load_tokenized_data(model.tokenizer)
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
        tokenizer=model.tokenizer,
        layer_index=layer,
        selected_features=torch.arange(0,1000),
        raw_dir= raw_features_path,
        n_random=5,
        max_examples=10000
    )

    for record in records:
        all_examples = record.examples
        if len(all_examples) < 500:
            continue
        top500 = all_examples[:500]

        sampling_technique = random.sample(top500, 20)+random.sample(all_examples[:-5], 15)+all_examples[-5:]
        
        explainer_inputs.append(
            ExplainerInput(
                train_examples=sampling_technique,
                record=record
            )
        )
        


client = get_client("local", "meta-llama/Meta-Llama-3-8B-Instruct", base_url="http://127.0.0.1:8000")

explainer = SimpleExplainer(client)
print("Running small llama")
explainer_out_dir = "saved_explanations/llama_small/"
asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
    )
)

