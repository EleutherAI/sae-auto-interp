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

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]

# Load model 
model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="cpu", dispatch=True,torch_dtype =torch.bfloat16)

# Load tokenized data
tokens = load_tokenized_data(model.tokenizer)

# Load features to explain
samples = get_samples(N_LAYERS=32,N_FEATURES=131072,N_SAMPLES=1000)

# Raw features contains locations
raw_features_path = "raw_features_llama"

explainer_inputs_10_100 = []
explainer_inputs_10_all = []
explainer_inputs_10_all_10 = []
explainer_inputs_20_100 = []
explainer_inputs_20_all = []
seed = 22
random.seed(seed)
        
for layer in layers:
    records = FeatureRecord.from_tensor(
        tokens,
        tokenizer=model.tokenizer,
        layer_index=layer,
        selected_features=samples[layer],
        raw_dir= raw_features_path,
        max_examples=2000
    )

    for record in records:
        all_examples = record.examples
        if len(all_examples) < 100:
            continue
        top100 = all_examples[:100]

        random_10_all = random.sample(all_examples, 10)
        random_20_all = random.sample(all_examples, 20)
        random_10_100 = random.sample(top100, 10)
        random_20_100 = random.sample(top100, 20)  
        random_10_all_10 = random.sample(all_examples, 10)+all_examples[:10]
        explainer_inputs_10_100.append(
            ExplainerInput(
                train_examples=random_10_all,
                record=record
            )
        )
        explainer_inputs_10_all.append(
            ExplainerInput(
                train_examples=random_10_100,
                record=record
            )
        )
        explainer_inputs_20_100.append(
            ExplainerInput(
                train_examples=random_20_all,
                record=record
            )
        )
        explainer_inputs_20_all.append(
            ExplainerInput(
                train_examples=random_20_100,
                record=record
            )
        )
        explainer_inputs_10_all_10.append(
            ExplainerInput(
                train_examples=random_10_all_10,
                record=record
            )
        )


client = get_client("local", "meta-llama/Meta-Llama-3-8B-Instruct", base_url="http://127.0.0.1:8000")

explainer = SimpleExplainer(client)
explainer_out_dir = "saved_explanations/llama_simple_top10_100/"

asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs_10_100,
        output_dir=explainer_out_dir,
    )
)
explainer_out_dir = "saved_explanations/llama_simple_top10_all/"
asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs_10_all,
        output_dir=explainer_out_dir,
    )
)

explainer_out_dir = "saved_explanations/llama_simple_top20_100/"
asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs_20_100,
        output_dir=explainer_out_dir,
    )
)

explainer_out_dir = "saved_explanations/llama_simple_top20_all/"
asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs_20_all,
        output_dir=explainer_out_dir,
    )
)

explainer_out_dir = "saved_explanations/llama_simple_top10_all_10/"
asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs_10_all_10,
        output_dir=explainer_out_dir,
    )
)

